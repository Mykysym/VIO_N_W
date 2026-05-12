"""Entry point: run monocular VO on a TUM VI sequence.

Usage: python run_vo.py --seq data/room2 --config configs/room2.yaml
"""
import argparse
import time
import traceback

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)

from src.utils.tum_vi_loader  import TUMVIDataset
from src.utils.trajectory_io  import save_tum_trajectory
from src.frontend.feature_detector import FeatureDetector
from src.frontend.feature_tracker  import FeatureTracker
from src.frontend.epipolar         import EpipolarGeometry
from src.backend.pnp_solver        import PnPSolver
from src.backend.bundle_adjustment import MotionOnlyBA


# ── helpers ────────────────────────────────────────────────────────────────────

def _triangulate_with_mask(K, R, t, pts1, pts2):
    """Triangulate and return (pts3d_cam1, valid_bool_mask).

    Points are in the first camera frame.  valid_mask marks points with
    positive depth in both cameras (chirality filter).  Unlike
    EpipolarGeometry.triangulate, this returns the mask so the caller can
    align the output with track IDs.
    """
    P1 = K @ np.hstack([np.eye(3),        np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts1.astype(np.float32).T,
        pts2.astype(np.float32).T,
    )
    w     = pts4d[3]
    pts3d = (pts4d[:3] / np.where(np.abs(w) > 1e-10, w, 1e-10)).T.astype(np.float64)
    depth1 = pts3d[:, 2]
    depth2 = (R @ pts3d.T + t.reshape(3, 1)).T[:, 2]
    valid  = (depth1 > 0) & (depth2 > 0)
    return pts3d, valid


def _T_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t).ravel()
    return T


# ── pipeline ───────────────────────────────────────────────────────────────────

def main():
    # ── 1. Args and config ────────────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",    required=True,  help="Path to TUM VI sequence")
    ap.add_argument("--config", required=True,  help="YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 0)
    np.random.seed(seed)

    vo_cfg        = cfg.get("vo", {})
    method        = vo_cfg.get("detector", "ORB")
    n_features    = vo_cfg.get("n_features", 1000)
    min_matches   = vo_cfg.get("min_matches", 30)
    ransac_thresh = float(vo_cfg.get("ransac_threshold", 1.0))
    seq_name      = cfg.get("seq_name", Path(args.seq).name)

    # ── 2. Dataset ────────────────────────────────────────────────────────
    ds = TUMVIDataset(args.seq, cam=cfg.get("cam", "cam0"))
    K  = ds.cam_calib.K
    print(f"\nK =\n{K}")
    print(f"\nT_cam_imu =\n{ds.cam_calib.T_cam_imu}\n")

    # ── Pipeline objects ──────────────────────────────────────────────────
    detector = FeatureDetector(method=method, n_features=n_features, seed=seed)
    tracker  = FeatureTracker(detector, min_tracks=min_matches)
    epipolar = EpipolarGeometry(K)
    solver   = PnPSolver(K, ransac_thresh=ransac_thresh * 4.0,
                         confidence=0.999, min_inliers=12)
    ba       = MotionOnlyBA(K, loss="huber", huber_delta=1.0)

    # ── State ─────────────────────────────────────────────────────────────
    trajectory:    list = []         # (timestamp_s, T_cw 4×4)
    landmark_map:  dict = {}         # track_id (int) → 3-D world point (3,)
    n_failures:    int  = 0
    t_wall_start         = time.time()
    T_prev               = np.eye(4, dtype=np.float64)   # updated each frame

    try:
        # ── 3. Initialisation — frames 0 and 1 ───────────────────────────
        frame_iter = ds.iter_frames()
        f0 = next(frame_iter)
        f1 = next(frame_iter)
        img0, img1 = f0["image"], f1["image"]

        # Init tracker on frame 0; track to frame 1 to get correspondences
        tracker.init(img0)
        prev_pts, curr_pts, ids = tracker.track(img1)

        # Essential matrix and pose recovery
        E, emask  = epipolar.estimate_essential(prev_pts, curr_pts,
                                                ransac_thresh=ransac_thresh)
        R, t, _   = epipolar.recover_pose(E, prev_pts, curr_pts, emask)

        # Triangulate inliers; need the validity mask to align with track IDs
        inlier_mask = emask.ravel().astype(bool)
        pts1_in = prev_pts[inlier_mask]
        pts2_in = curr_pts[inlier_mask]
        ids_in  = ids[inlier_mask]

        pts3d_raw, valid = _triangulate_with_mask(K, R, t, pts1_in, pts2_in)
        pts3d_good = pts3d_raw[valid]

        # Scale: median depth in cam-0 frame (= world frame since T_0 = I)
        scale = epipolar.compute_scale(pts3d_good)

        T_0 = np.eye(4, dtype=np.float64)
        T_1 = _T_from_Rt(R, t.ravel() * scale)

        # Initial map: IDs → world points (T_0 = I  ⟹  cam-0 = world)
        pts3d_scaled = pts3d_good * scale
        for tid, pt in zip(ids_in[valid], pts3d_scaled):
            landmark_map[int(tid)] = pt

        trajectory.append((f0["timestamp"], T_0))
        trajectory.append((f1["timestamp"], T_1))
        T_prev = T_1

        print(f"[Init]  scale={scale:.4f} m  "
              f"landmarks={len(landmark_map)}  "
              f"inliers={int(inlier_mask.sum())}")

        # ── 4. Tracking phase — frames 2 … N ─────────────────────────────
        for frame in frame_iter:
            img = frame["image"]
            idx = frame["index"]
            ts  = frame["timestamp"]

            # a. Track
            prev_pts_t, curr_pts_t, ids_t = tracker.track(img)

            # b. Split into known-landmark and new-track subsets
            known_mask = np.array([int(tid) in landmark_map for tid in ids_t],
                                  dtype=bool)
            n_known = int(known_mask.sum())

            if n_known < solver.min_inliers:
                n_failures += 1
                print(f"[Frame {idx:4d}] SKIP  — only {n_known} landmarks visible")
                trajectory.append((ts, T_prev))
                continue

            pts3d_k   = np.array([landmark_map[int(tid)]
                                  for tid in ids_t[known_mask]], dtype=np.float64)
            curr_pts_k = curr_pts_t[known_mask]

            # c. PnP
            try:
                T_est, pnp_inliers = solver.solve(pts3d_k, curr_pts_k,
                                                  initial_pose=T_prev)
            except RuntimeError as exc:
                n_failures += 1
                print(f"[Frame {idx:4d}] PnP FAILED — {exc}")
                trajectory.append((ts, T_prev))
                continue

            # d. Motion-only BA
            T_refined, rms = ba.optimise(T_est,
                                         pts3d_k[pnp_inliers],
                                         curr_pts_k[pnp_inliers])

            # e. Triangulate new tracks
            new_mask = ~known_mask
            n_prev = len(prev_pts_t)
            # Re-detected features at index >= n_prev have no prev-frame position
            triag_mask = new_mask.copy()
            triag_mask[n_prev:] = False
            if triag_mask.sum() >= 5:
                T_rel   = T_refined @ np.linalg.inv(T_prev)
                R_rel   = T_rel[:3, :3]
                t_rel   = T_rel[:3,  3]
                if np.linalg.norm(t_rel) > 1e-3:
                    new3d, new_valid = _triangulate_with_mask(
                        K, R_rel, t_rel,
                        prev_pts_t[triag_mask[:n_prev]],
                        curr_pts_t[triag_mask],
                    )
                    if new_valid.any():
                        # Transform from prev-cam frame to world frame
                        T_prev_wc = np.linalg.inv(T_prev)
                        pts_world = (
                            T_prev_wc[:3, :3] @ new3d[new_valid].T
                            + T_prev_wc[:3, 3:]
                        ).T
                        for tid, pt in zip(ids_t[triag_mask][new_valid], pts_world):
                            landmark_map[int(tid)] = pt

            trajectory.append((ts, T_refined))
            T_prev = T_refined

            if idx % 50 == 0:
                print(f"[Frame {idx:4d}]  "
                      f"tracked={tracker.n_tracked:4d}  "
                      f"map={len(landmark_map):5d}  "
                      f"BA_rms={rms:.3f} px  "
                      f"failures={n_failures}")

    except Exception:
        print("\n[ERROR] Pipeline crashed — saving partial results.")
        traceback.print_exc()

    # ── 5. Save results ───────────────────────────────────────────────────
    elapsed   = time.time() - t_wall_start
    n_frames  = len(trajectory)

    print(f"\nTotal runtime    : {elapsed:.1f} s")
    print(f"Frames processed : {n_frames}")
    if n_frames > 0:
        print(f"Mean ms/frame    : {1000.0 * elapsed / n_frames:.1f} ms")
    print(f"PnP failures     : {n_failures}")

    if not trajectory:
        print("[WARN] No trajectory to save.")
        return

    out_traj_dir = Path("results/trajectories")
    out_plot_dir = Path("results/plots/trajectories")
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    timestamps_s = [ts for ts, _ in trajectory]
    poses_cw     = [T  for _, T  in trajectory]

    # TUM format expects world-from-camera; invert our camera-from-world poses
    poses_wc = [np.linalg.inv(T) for T in poses_cw]
    traj_path = out_traj_dir / f"vo_{seq_name}.txt"
    save_tum_trajectory(poses_wc, timestamps_s, str(traj_path))

    # Camera centres in world frame for the plot
    positions = np.array([T[:3, 3] for T in poses_wc])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(positions[:, 0], positions[:, 2],
            linewidth=1.0, color="steelblue", label="VO estimate")
    ax.scatter(positions[0,  0], positions[0,  2],
               c="green", s=60, zorder=5, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 2],
               c="red",   s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z — forward (m)")
    ax.set_title(f"VO trajectory — {seq_name}")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_plot_dir / f"vo_{seq_name}.png"
    plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved       → {plot_path}")


if __name__ == "__main__":
    main()
