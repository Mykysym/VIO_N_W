"""Entry point: run VIO (VO + IMU) on a TUM VI sequence.

Usage: python run_vio.py --seq data/room2 --config configs/room2.yaml

Pipeline:
  Phase 1 – VO bootstrap to accumulate SfM poses + IMU segments.
  Phase 2 – VIOInitializer recovers metric scale, gravity, velocities, bias.
  Phase 3 – Tracking: PnP keyframes fed to SlidingWindowOptimizer;
             non-keyframes use IMU-only propagation.
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

from src.utils.tum_vi_loader        import TUMVIDataset
from src.utils.trajectory_io        import save_tum_trajectory
from src.frontend.feature_detector  import FeatureDetector
from src.frontend.feature_tracker   import FeatureTracker
from src.frontend.epipolar          import EpipolarGeometry
from src.backend.pnp_solver         import PnPSolver
# from src.backend.bundle_adjustment  import MotionOnlyBA
from src.backend.imu_preintegration import IMUPreintegration, Exp
from src.backend.vio_initializer    import VIOInitializer
from src.backend.sliding_window     import SlidingWindowOptimizer


# ── geometry helpers ───────────────────────────────────────────────────────────

def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t).ravel()
    return T


def _triangulate_with_mask(K, R, t, pts1, pts2):
    """Triangulate point pairs in camera-1 frame; return (pts3d, valid_mask).

    valid_mask is True where depth is positive in both cameras (chirality).
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
    return pts3d, (depth1 > 0) & (depth2 > 0)


def _T_wb_from_T_cw(T_cw: np.ndarray, T_cam_imu: np.ndarray) -> np.ndarray:
    """T_{world←body} from T_{cam←world}."""
    return np.linalg.inv(T_cw) @ T_cam_imu


def _T_cw_from_T_wb(T_wb: np.ndarray, T_cam_imu: np.ndarray) -> np.ndarray:
    """T_{cam←world} from T_{world←body}."""
    return T_cam_imu @ np.linalg.inv(T_wb)


def _integrate_rotation_only(imu_meas: list, b_g: np.ndarray) -> np.ndarray:
    """Gyro-only SO(3) integration over an IMU segment → delta_R (body frame)."""
    R = np.eye(3, dtype=np.float64)
    for k in range(1, len(imu_meas)):
        dt = float(imu_meas[k]['t']) - float(imu_meas[k - 1]['t'])
        if dt <= 0.0:
            continue
        omega = (np.array([imu_meas[k-1]['wx'], imu_meas[k-1]['wy'],
                            imu_meas[k-1]['wz']])
                 - b_g)
        R = R @ Exp(omega * dt)
    return R


def _rotation_compensated_parallax(kf_pts_dict: dict,
                                    curr_pts_t:  np.ndarray,
                                    ids_t:       np.ndarray,
                                    R_cam_delta: np.ndarray,
                                    K:           np.ndarray) -> float:
    """Mean rotation-compensated parallax (px) between current frame and last KF.

    Reference pixel is un-projected, rotated by R_{c_cur←c_kf}, re-projected,
    and pixel distance to current observation is measured.
    """
    if not kf_pts_dict:
        return 0.0

    K_inv = np.linalg.inv(K)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    parallax_list = []

    for i, tid in enumerate(ids_t):
        ref = kf_pts_dict.get(int(tid))
        if ref is None:
            continue
        u_ref, v_ref = float(ref[0]), float(ref[1])
        u_cur, v_cur = float(curr_pts_t[i, 0]), float(curr_pts_t[i, 1])

        p_n   = K_inv @ np.array([u_ref, v_ref, 1.0])
        p_rot = R_cam_delta @ p_n
        z     = p_rot[2]
        if abs(z) < 1e-6:
            continue
        u_rot = fx * p_rot[0] / z + cx
        v_rot = fy * p_rot[1] / z + cy
        parallax_list.append(np.sqrt((u_cur - u_rot) ** 2 + (v_cur - v_rot) ** 2))

    return float(np.mean(parallax_list)) if parallax_list else 0.0


def _prop_imu(T_wb_kf:  np.ndarray,
              v_kf:     np.ndarray,
              b_g:      np.ndarray,
              b_a:      np.ndarray,
              gravity:  np.ndarray,
              imu_meas: list,
              imu_calib) -> tuple:
    """Propagate IMU state forward from last keyframe.

    Returns (T_wb_new, v_new). Falls back to KF state if <2 IMU samples.
    """
    if len(imu_meas) < 2:
        return T_wb_kf.copy(), v_kf.copy()

    p = IMUPreintegration(b_g, b_a, imu_calib)
    p.integrate_imu_segment(imu_meas)
    dt = p.dt_sum

    R_wb = T_wb_kf[:3, :3]
    p_w  = T_wb_kf[:3,  3]

    R_wb_new = R_wb @ p.delta_R
    v_new    = v_kf + gravity * dt + R_wb @ p.delta_v
    p_w_new  = p_w + v_kf * dt + 0.5 * gravity * dt ** 2 + R_wb @ p.delta_p

    T_wb_new = np.eye(4, dtype=np.float64)
    T_wb_new[:3, :3] = R_wb_new
    T_wb_new[:3,  3] = p_w_new
    return T_wb_new, v_new


# ── main pipeline ──────────────────────────────────────────────────────────────

def main():
    # ── 1. Setup ───────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="Run VIO (visual-inertial odometry) on a TUM VI sequence.")
    ap.add_argument("--seq",    required=True, help="Path to TUM VI sequence dir")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 0)
    np.random.seed(seed)

    vo_cfg  = cfg.get("vo",  {})
    vio_cfg = cfg.get("vio", {})

    method             = vo_cfg.get("detector",          "ORB")
    n_features         = vo_cfg.get("n_features",         1000)
    min_matches        = vo_cfg.get("min_matches",          30)
    min_tracks         = vo_cfg.get("min_tracks", max(min_matches, 30))
    ransac_thresh      = float(vo_cfg.get("ransac_threshold", 1.0))
    seq_name           = cfg.get("seq_name", Path(args.seq).name)

    window_size        = int(vio_cfg.get("window_size",          5))
    min_init_frames    = int(vio_cfg.get("min_init_frames",     10))
    max_init_frames    = int(vio_cfg.get("max_init_frames",     80))
    kf_parallax_thresh = float(vio_cfg.get("kf_parallax_thresh", 30.0))
    kf_min_tracks      = int(vio_cfg.get("kf_min_tracks",       80))
    kf_max_interval    = int(vio_cfg.get("kf_max_interval",      5))
    g_cfg              = vio_cfg.get("gravity", [0.0, 0.0, -9.81])
    g_prior            = np.array(g_cfg, dtype=np.float64)

    # IMU / visual information balance.
    # For a 50 ms interval the IMU information per DOF is ~6e9 while the visual
    # information per pose DOF is ~890 000 (200 landmarks at 1.9 m depth, 1.5 px
    # sigma).  The 6700× imbalance makes the Gauss-Newton Hessian near-singular
    # in the directions not covered by visual factors, causing the SW to assign
    # physically impossible velocities within 2–3 keyframes.
    # Scaling the IMU information matrix down to ~1–10× visual restores balance.
    _imu_weight = float(vio_cfg.get("imu_weight", 1.0))
    if abs(_imu_weight - 1.0) > 1e-9:
        from src.backend import imu_factor as _imu_factor_mod
        _orig_info = _imu_factor_mod.IMUFactor.information_matrix
        def _scaled_info(self, _w=_imu_weight, _orig=_orig_info):
            return _orig(self) * _w
        _imu_factor_mod.IMUFactor.information_matrix = _scaled_info
        print(f"[VIO] IMU information matrix scaled by {_imu_weight}")

    ds        = TUMVIDataset(args.seq, cam=cfg.get("cam", "cam0"))
    K         = ds.cam_calib.K
    T_cam_imu = ds.cam_calib.T_cam_imu
    R_ci      = T_cam_imu[:3, :3]

    print(f"\nK =\n{K}")
    print(f"\nT_cam_imu =\n{T_cam_imu}\n")

    # Frontend (VO — frozen, never modified)
    detector = FeatureDetector(method=method, n_features=n_features, seed=seed)
    tracker  = FeatureTracker(detector, min_tracks=min_tracks)
    epipolar = EpipolarGeometry(K)
    solver   = PnPSolver(K, ransac_thresh=ransac_thresh * 4.0,
                         confidence=0.999, min_inliers=8)
    # ba       = MotionOnlyBA(K, loss="huber", huber_delta=1.0)

    # IMU backend
    g_mag    = float(np.linalg.norm(g_prior))
    vio_init = VIOInitializer(K, T_cam_imu, ds.imu_calib,
                               gravity_magnitude=g_mag if g_mag > 1.0 else 9.81,
                               min_frames=min_init_frames)
    sw       = SlidingWindowOptimizer(K, T_cam_imu, ds.imu_calib,
                                      window_size=window_size,
                                      gravity=g_prior)

    # Output paths (trajectory file written at end)
    out_traj_dir = Path("results/trajectories")
    out_plot_dir = Path("results/plots/trajectories")
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_traj_dir / f"vio_{seq_name}.txt"

    # Running state
    trajectory:   list = []    # (timestamp_s, T_cw 4×4)
    landmark_map: dict = {}    # track_id → 3-D world point (metric)
    n_failures        = 0
    n_keyframes       = 0
    init_frame_idx    = -1
    t_wall_start      = time.time()
    T_prev            = np.eye(4, dtype=np.float64)
    motion_hist: list = []     # recent inter-frame camera displacements

    # Pre-init SfM buffer: list of (T_cw, ts, imu_since_last, ids, pts2d)
    sfm_buffer: list = []

    # Tracking-phase state
    b_g_cur         = np.zeros(3)
    b_a_cur         = np.zeros(3)
    gravity         = g_prior.copy()
    last_kf_Twb     = np.eye(4, dtype=np.float64)
    last_kf_v       = np.zeros(3)
    kf_imu:    list = []    # IMU accumulator since last keyframe
    kf_pts_dict: dict = {}  # track_id → (u,v) at last keyframe
    frames_since_kf = 0
    sw_cost         = 0.0
    initialized     = False

    try:
        frame_iter = ds.iter_frames()

        # ── Bootstrap: advance until a reliable E-matrix ─────────────────────
        # Frames 0→1 (50 ms) can produce a near-zero baseline, causing RANSAC to
        # return a degenerate (~180°) rotation whose chirality check produces zero
        # valid triangulations.  We therefore advance frame-by-frame, always
        # estimating E between f0 and the current candidate, until the recovered
        # rotation is physically plausible (< 30°) AND we have enough well-
        # triangulated landmarks.  All IMU since f0 is accumulated so no data
        # is lost.  If no good candidate is found in MAX_BOOTSTRAP_FRAMES we
        # fall back unconditionally so the pipeline can still run.
        _MAX_BOOTSTRAP_FRAMES   = 50   # give up after 2.5 s @ 20 Hz
        _MIN_BOOTSTRAP_ROT_DEG  = 30.0 # reject rotations >= this (degenerate)
        _MIN_BOOTSTRAP_LANDMARKS = 12  # minimum valid triangulated map points

        f0 = next(frame_iter)
        T_0 = np.eye(4, dtype=np.float64)
        tracker.init(f0["image"])

        trajectory.append((f0["timestamp"], T_0))
        sfm_buffer.append((T_0, f0["timestamp"], f0["imu_since_last"],
                            np.array([], dtype=np.int64),
                            np.empty((0, 2), dtype=np.float32)))
        vio_init.add_frame(T_0, f0["timestamp"], f0["imu_since_last"])
        T_prev = T_0

        # Accumulated IMU from f0 to the current candidate (so preintegration
        # between the f0 keyframe and the bootstrap keyframe is complete).
        _accum_imu: list = []
        _bootstrap_done = False

        for _f_cand in frame_iter:
            _prev_c, _curr_c, _ids_c = tracker.track(_f_cand["image"])
            _accum_imu.extend(_f_cand["imu_since_last"])

            _n_tried  = _f_cand["index"]
            _fallback = (_n_tried >= _MAX_BOOTSTRAP_FRAMES)

            # Use CONSECUTIVE prev→curr (tracker always has full coverage).
            # Since every pre-bootstrap frame is frozen at T_0=identity, the
            # world frame = f0's camera frame, so consecutive-frame E-matrix
            # produces the correct absolute T_1 in world coordinates.
            if len(_prev_c) >= min_matches:
                try:
                    _E, _em = epipolar.estimate_essential(_prev_c, _curr_c,
                                                          ransac_thresh=ransac_thresh)
                    _R, _t, _ = epipolar.recover_pose(_E, _prev_c, _curr_c, _em)
                    _rot_deg = float(np.degrees(np.arccos(
                        np.clip((np.trace(_R) - 1.0) / 2.0, -1.0, 1.0))))
                    _im = _em.ravel().astype(bool)
                    _p1, _p2 = _prev_c[_im], _curr_c[_im]
                    _pi      = _ids_c[_im]
                    _pts3d_r, _valid = _triangulate_with_mask(K, _R, _t, _p1, _p2)
                    _n_valid = int(_valid.sum())

                    _good = (_rot_deg < _MIN_BOOTSTRAP_ROT_DEG
                             and _n_valid >= _MIN_BOOTSTRAP_LANDMARKS)

                    if _good or _fallback:
                        _sfm_scale = 1.0
                        if _valid.any():
                            _med = epipolar.compute_scale(_pts3d_r[_valid])
                            _sfm_scale = 1.0 / _med if _med > 1e-6 else 1.0
                        T_1      = _T_from_Rt(_R, _t.ravel() * _sfm_scale)
                        _p3d_sc  = _pts3d_r[_valid] * _sfm_scale
                        _ids_val = _pi[_valid]
                        for _tid2, _pt in zip(_ids_val, _p3d_sc):
                            landmark_map[int(_tid2)] = _pt

                        trajectory.append((_f_cand["timestamp"], T_1))
                        T_prev = T_1
                        sfm_buffer.append((T_1, _f_cand["timestamp"],
                                           _accum_imu,
                                           _ids_val, _p2[_valid]))
                        # Reset vio_init so it only sees poses after bootstrap.
                        vio_init = VIOInitializer(
                            K, T_cam_imu, ds.imu_calib,
                            gravity_magnitude=g_mag if g_mag > 1.0 else 9.81,
                            min_frames=min_init_frames)
                        vio_init.add_frame(T_0, f0["timestamp"],
                                           f0["imu_since_last"])
                        vio_init.add_frame(T_1, _f_cand["timestamp"],
                                           _accum_imu)

                        print(f"[Bootstrap]  frame={_f_cand['index']}  "
                              f"rot={_rot_deg:.1f}deg  "
                              f"sfm_scale={_sfm_scale:.4f}  "
                              f"landmarks={len(landmark_map)}  "
                              f"{'(fallback)' if _fallback else ''}")
                        _bootstrap_done = True
                        break
                except Exception:
                    pass

            if _fallback and not _bootstrap_done:
                print(f"[WARN] Bootstrap gave up at frame {_n_tried}; "
                      "landmark_map may be empty.")
                break

            # Buffer frame with T_0 (pre-bootstrap, camera at rest estimate).
            trajectory.append((_f_cand["timestamp"], T_0.copy()))
            sfm_buffer.append((T_0.copy(), _f_cand["timestamp"],
                                _f_cand["imu_since_last"],
                                np.array([], dtype=np.int64),
                                np.empty((0, 2), dtype=np.float32)))

        # ── Main frame loop ───────────────────────────────────────────────────
        for frame in frame_iter:
            img = frame["image"]
            idx = frame["index"]
            ts  = frame["timestamp"]
            imu = frame["imu_since_last"]

            # a. Track features (shared by both phases)
            prev_pts_t, curr_pts_t, ids_t = tracker.track(img)
            n_tracked = tracker.n_tracked

            known_mask = np.array([int(tid) in landmark_map for tid in ids_t],
                                  dtype=bool)
            n_known = int(known_mask.sum())

            # ═════════════════════════════════════════════════════════════════
            # PHASE 1 — PRE-INITIALIZATION (VO bootstrap)
            # ═════════════════════════════════════════════════════════════════
            if not initialized:
                # Timeout: fall through with zero-velocity state and gravity prior
                if idx > max_init_frames:
                    print(f"[WARN] VIO init timed out at frame {idx} — "
                          f"proceeding with VO poses and gravity prior.")
                    initialized     = True
                    init_frame_idx  = idx
                    gravity         = g_prior.copy()
                    b_g_cur         = np.zeros(3)
                    b_a_cur         = np.zeros(3)
                    last_kf_Twb     = _T_wb_from_T_cw(T_prev, T_cam_imu)
                    last_kf_v       = np.zeros(3)
                    kf_imu          = []
                    kf_pts_dict     = {}
                    frames_since_kf = 0
                    continue

                # b. Estimate pose with PnP (same as run_vo.py)
                T_cur = T_prev
                if n_known >= solver.min_inliers:
                    pts3d_k    = np.array([landmark_map[int(tid)]
                                           for tid in ids_t[known_mask]])
                    curr_pts_k = curr_pts_t[known_mask]
                    try:
                        T_est, pnp_inliers = solver.solve(pts3d_k, curr_pts_k,
                                                          initial_pose=T_prev)
                        T_cur = T_est          # BA disabled
                        # T_cur, _ = ba.optimise(T_est,
                        #                        pts3d_k[pnp_inliers],
                        #                        curr_pts_k[pnp_inliers])
                        # Triangulate new tracks to grow the map
                        new_mask   = ~known_mask
                        n_prev_pts = len(prev_pts_t)
                        tri_mask   = new_mask.copy()
                        tri_mask[n_prev_pts:] = False
                        if tri_mask.sum() >= 5:
                            T_rel = T_cur @ np.linalg.inv(T_prev)
                            R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
                            if np.linalg.norm(t_rel) > 1e-3:
                                new3d, nv = _triangulate_with_mask(
                                    K, R_rel, t_rel,
                                    prev_pts_t[tri_mask[:n_prev_pts]],
                                    curr_pts_t[tri_mask])
                                if nv.any():
                                    T_wc = np.linalg.inv(T_prev)
                                    pw   = (T_wc[:3, :3] @ new3d[nv].T
                                            + T_wc[:3, 3:]).T
                                    for tid, pt in zip(ids_t[tri_mask][nv], pw):
                                        landmark_map[int(tid)] = pt
                        cam_prev = np.linalg.inv(T_prev)[:3, 3]
                        cam_cur  = np.linalg.inv(T_cur)[:3, 3]
                        motion_hist.append(float(np.linalg.norm(cam_cur - cam_prev)))
                        if len(motion_hist) > 20:
                            motion_hist.pop(0)
                    except RuntimeError as exc:
                        n_failures += 1
                        print(f"[Pre-init {idx:4d}] PnP FAILED — {exc}")
                else:
                    # Recovery: E-matrix + motion-history scale
                    n_prev_pts = len(prev_pts_t)
                    if n_prev_pts >= min_matches and len(motion_hist) >= 5:
                        try:
                            avg_d = float(np.median(motion_hist))
                            if avg_d > 1e-8:
                                E_r, em_r = epipolar.estimate_essential(
                                    prev_pts_t, curr_pts_t,
                                    ransac_thresh=ransac_thresh)
                                R_r, t_r, _ = epipolar.recover_pose(
                                    E_r, prev_pts_t, curr_pts_t, em_r)
                                t_sc = t_r.ravel() * avg_d
                                in_r = em_r.ravel().astype(bool)
                                n3d_r, v_r = _triangulate_with_mask(
                                    K, R_r, t_sc,
                                    prev_pts_t[in_r], curr_pts_t[in_r])
                                if v_r.sum() >= solver.min_inliers:
                                    dep = n3d_r[v_r][:, 2]
                                    dok = (dep > 0.01) & (dep < 20.0)
                                    if dok.sum() >= solver.min_inliers:
                                        T_wc = np.linalg.inv(T_prev)
                                        pw   = (T_wc[:3, :3] @ n3d_r[v_r][dok].T
                                                + T_wc[:3, 3:]).T
                                        for tid, pt in zip(
                                                ids_t[:n_prev_pts][in_r][v_r][dok],
                                                pw):
                                            landmark_map[int(tid)] = pt
                        except Exception:
                            pass
                    n_failures += 1

                # c. Buffer VO pose for VIOInitializer
                sfm_buffer.append((T_cur, ts, imu,
                                    ids_t[known_mask],
                                    curr_pts_t[known_mask]))
                trajectory.append((ts, T_cur))
                T_prev = T_cur

                # c. Feed to VIOInitializer; check if initialization succeeded
                init_done = vio_init.add_frame(T_cur, ts, imu)

                if init_done:
                    states  = vio_init.get_initial_states()
                    s       = float(states["scale"])
                    gravity = states["gravity"].copy()
                    b_g_cur = states["b_g"].copy()
                    b_a_cur = states["b_a"].copy()

                    print(f"[VIO] Initialized at frame {idx}, scale={s:.3f}")
                    print(f"[VIO]   gravity = {gravity.round(3)}, |v_last| = {np.linalg.norm(last_kf_v):.3f} m/s") #temp

                    init_frame_idx = idx
                    initialized    = True

                    # Apply metric scale to all existing landmarks and poses
                    for tid in list(landmark_map.keys()):
                        landmark_map[tid] = landmark_map[tid] * s

                    for i, (ts_k, T_k) in enumerate(trajectory):
                        T_sc        = T_k.copy()
                        T_sc[:3, 3] *= s
                        trajectory[i] = (ts_k, T_sc)

                    T_prev = trajectory[-1][1]

                    # Rebuild the SW with the recovered gravity vector
                    sw = SlidingWindowOptimizer(K, T_cam_imu, ds.imu_calib,
                                               window_size=window_size,
                                               gravity=gravity)

                    # Do NOT seed the SW from the init buffer.
                    # Seeding all 30 buffered near-stationary frames creates 25
                    # consecutive Schur-complement marginalizations; the resulting
                    # prior from v≈0 frames conflicts with the first active-motion
                    # keyframe's IMU constraint and drives velocity to ~18 m/s.
                    # Starting empty lets the SW build up cleanly from the first
                    # tracking keyframe, where all factors are well-conditioned.
                    n_keyframes += len(sfm_buffer)
                    for k, (T_cw_buf, ts_buf, imu_buf,
                             ids_buf, pts2d_buf) in enumerate(sfm_buffer):
                        break   # skip all: SW starts empty
                        T_cw_m        = T_cw_buf.copy()
                        T_cw_m[:3, 3] *= s
                        v_k = (states["velocities"][k].copy()
                               if k < len(states["velocities"]) else np.zeros(3))
                        sw_p3d, sw_p2d = [], []
                        for tid, pt2d in zip(ids_buf, pts2d_buf):
                            if int(tid) in landmark_map:
                                sw_p3d.append(landmark_map[int(tid)])
                                sw_p2d.append(pt2d)
                        sw.add_keyframe(T_cw_m, v_k, b_g_cur, b_a_cur,
                                        ts_buf, sw_p3d, sw_p2d, imu_buf)

                    # Synchronise tracking state with the last init pose
                    init_poses = states["poses"]   # T_wb per frame (metric)
                    if init_poses:
                        last_kf_Twb = init_poses[-1].copy()
                        T_prev      = _T_cw_from_T_wb(last_kf_Twb, T_cam_imu)
                        if trajectory:
                            trajectory[-1] = (trajectory[-1][0], T_prev)
                    else:
                        last_kf_Twb = _T_wb_from_T_cw(T_prev, T_cam_imu)

                    last_kf_v = (states["velocities"][-1].copy()
                                 if states["velocities"] else np.zeros(3))
                    # Sanity-clamp: typical hand-held camera speed < 5 m/s.
                    # The VIO init linear system can be ill-conditioned for short
                    # windows, returning absurd velocity estimates that cause
                    # IMU propagation to drift several metres per frame.
                    _v_mag = float(np.linalg.norm(last_kf_v))
                    if _v_mag > 5.0:
                        print(f"[VIO] Init velocity clamped "
                              f"({_v_mag:.1f} → 5.0 m/s)")
                        last_kf_v = last_kf_v * (5.0 / _v_mag)
                    kf_imu          = []
                    kf_pts_dict     = {int(tid): pt
                                       for tid, pt in zip(ids_t, curr_pts_t)}
                    frames_since_kf = 0

                continue

            # ═════════════════════════════════════════════════════════════════
            # PHASE 2 — TRACKING (post-initialization)
            # ═════════════════════════════════════════════════════════════════

            # Accumulate IMU since last keyframe
            kf_imu.extend(imu)
            frames_since_kf += 1

            # d. IMU propagation: cheap pose for non-KF and PnP initial guess
            T_wb_prop, v_prop = _prop_imu(last_kf_Twb, last_kf_v,
                                           b_g_cur, b_a_cur, gravity,
                                           kf_imu, ds.imu_calib)
            T_cw_prop = _T_cw_from_T_wb(T_wb_prop, T_cam_imu)

            # b. Keyframe decision
            #    R_{c_cur←c_kf} = R_ci @ delta_R_body.T @ R_ci.T
            delta_R_body = _integrate_rotation_only(kf_imu, b_g_cur)
            R_cam_delta  = R_ci @ delta_R_body.T @ R_ci.T
            parallax     = _rotation_compensated_parallax(
                kf_pts_dict, curr_pts_t, ids_t, R_cam_delta, K)

            need_kf = (parallax        > kf_parallax_thresh
                       or n_tracked    < kf_min_tracks
                       or frames_since_kf >= kf_max_interval)

            if need_kf:
                # c. Keyframe: PnP + motion-only BA + sliding-window optimization
                n_keyframes += 1
                T_cur = T_cw_prop   # IMU prediction as fallback
                v_cur = v_prop

                if n_known >= solver.min_inliers:
                    pts3d_k    = np.array([landmark_map[int(tid)]
                                           for tid in ids_t[known_mask]])
                    curr_pts_k = curr_pts_t[known_mask]
                    try:
                        T_est, pnp_inliers = solver.solve(pts3d_k, curr_pts_k,
                                                          initial_pose=T_prev)
                        T_cur = T_est          # BA disabled
                        # T_cur, _ = ba.optimise(T_est,
                        #                        pts3d_k[pnp_inliers],
                        #                        curr_pts_k[pnp_inliers])
                        v_cur = v_prop

                        # Update map with newly triangulated tracks
                        new_mask   = ~known_mask
                        n_prev_pts = len(prev_pts_t)
                        tri_mask   = new_mask.copy()
                        tri_mask[n_prev_pts:] = False
                        if tri_mask.sum() >= 5:
                            T_rel = T_cur @ np.linalg.inv(T_prev)
                            R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
                            if np.linalg.norm(t_rel) > 1e-3:
                                try:
                                    new3d, nv = _triangulate_with_mask(
                                        K, R_rel, t_rel,
                                        prev_pts_t[tri_mask[:n_prev_pts]],
                                        curr_pts_t[tri_mask])
                                    if nv.any():
                                        dep = new3d[nv][:, 2]
                                        dok = (dep > 0.01) & (dep < 50.0)
                                        if dok.any():
                                            T_wc = np.linalg.inv(T_prev)
                                            pw   = (T_wc[:3, :3] @ new3d[nv][dok].T
                                                    + T_wc[:3, 3:]).T
                                            for tid, pt in zip(
                                                    ids_t[tri_mask][nv][dok], pw):
                                                landmark_map[int(tid)] = pt
                                except Exception:
                                    pass

                        # Assemble landmark data for the SW (PnP inliers only)
                        ids_kf   = ids_t[known_mask][pnp_inliers]
                        pts2d_kf = curr_pts_k[pnp_inliers]
                        sw_p3d, sw_p2d = [], []
                        for tid, pt2d in zip(ids_kf, pts2d_kf):
                            if int(tid) in landmark_map:
                                sw_p3d.append(landmark_map[int(tid)])
                                sw_p2d.append(pt2d)

                        sw.add_keyframe(T_cur, v_cur, b_g_cur, b_a_cur,
                                        ts, sw_p3d, sw_p2d, kf_imu)

                        sw_res  = sw.optimize(n_iterations=5)
                        sw_cost = sw_res["final_cost"]
                        if not sw_res["converged"]:
                            print(f"[Frame {idx:4d}] SW did not converge "
                                  f"(|δx|={sw_res['delta_norm']:.2e}); "
                                  f"continuing with current state.")

                        # Accept SW output only if numerically sane.
                        # Guard on both velocity magnitude AND position drift
                        # from the PnP estimate: a diverging SW will move the
                        # camera far from where the visual measurement placed
                        # it, which we treat as a rejection signal.
                        T_sw  = sw.get_latest_pose()
                        v_sw  = sw.get_latest_velocity().copy()
                        p_sw  = np.linalg.inv(T_sw)[:3, 3]
                        p_pnp = np.linalg.inv(T_est)[:3, 3]
                        pos_drift = float(np.linalg.norm(p_sw - p_pnp))
                        if (np.isfinite(sw_cost)
                                and np.linalg.norm(v_sw) < 20.0
                                and pos_drift < 2.0
                                and np.all(np.isfinite(T_sw))):
                            T_cur = T_sw
                            v_cur = v_sw
                            b_sw  = sw.get_latest_bias()
                            b_g_cur = b_sw["b_g"]
                            b_a_cur = b_sw["b_a"]
                        else:
                            print(f"[Frame {idx:4d}] SW output rejected "
                                  f"(cost={sw_cost:.2e}, "
                                  f"|v|={np.linalg.norm(v_sw):.1f} m/s, "
                                  f"drift={pos_drift:.2f} m); "
                                  f"keeping PnP estimate.")
                            # Reset the SW window so a corrupted prior from
                            # this bad state doesn't poison future keyframes.
                            sw = SlidingWindowOptimizer(
                                K, T_cam_imu, ds.imu_calib,
                                window_size=window_size, gravity=gravity)
                            # Estimate velocity from consecutive PnP positions
                            # (finite-difference) — robust to IMU scale errors.
                            _dt_kf = ((float(kf_imu[-1]['t'])
                                       - float(kf_imu[0]['t']))
                                      if len(kf_imu) >= 2 else 0.25)
                            if _dt_kf > 1e-4:
                                _p_p = np.linalg.inv(T_prev)[:3, 3]
                                _p_c = np.linalg.inv(T_est)[:3, 3]
                                v_cur = (_p_c - _p_p) / _dt_kf
                            else:
                                v_cur = np.zeros(3, dtype=np.float64)
                            _vm = float(np.linalg.norm(v_cur))
                            if _vm > 5.0:
                                v_cur *= 5.0 / _vm

                    except RuntimeError as exc:
                        n_failures += 1
                        print(f"[Frame {idx:4d}] PnP FAILED — {exc}; "
                              f"using IMU propagated pose.")
                else:
                    n_failures += 1
                    print(f"[Frame {idx:4d}] SKIP — only {n_known} landmarks; "
                          f"using IMU propagated pose.")
                    # Re-triangulate using IMU-propagated pose so the map
                    # can recover rather than staying empty indefinitely.
                    n_prev_pts = len(prev_pts_t)
                    if n_prev_pts >= min_matches and len(motion_hist) >= 3:
                        avg_d = float(np.median(motion_hist))
                        if avg_d > 1e-4:
                            T_rel = T_cur @ np.linalg.inv(T_prev)
                            R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
                            if np.linalg.norm(t_rel) > 1e-4:
                                try:
                                    n3d_r, v_r = _triangulate_with_mask(
                                        K, R_rel, t_rel,
                                        prev_pts_t[:n_prev_pts],
                                        curr_pts_t[:n_prev_pts])
                                    if v_r.sum() >= solver.min_inliers:
                                        dep = n3d_r[v_r][:, 2]
                                        dok = (dep > 0.01) & (dep < 50.0)
                                        if dok.sum() >= solver.min_inliers:
                                            T_wc = np.linalg.inv(T_prev)
                                            pw   = (T_wc[:3, :3] @ n3d_r[v_r][dok].T
                                                    + T_wc[:3, 3:]).T
                                            for tid, pt in zip(
                                                    ids_t[:n_prev_pts][v_r][dok], pw):
                                                landmark_map[int(tid)] = pt
                                except Exception:
                                    pass

                # Update motion history from IMU-propagated displacement too.
                cam_p = np.linalg.inv(T_prev)[:3, 3]
                cam_c = np.linalg.inv(T_cur)[:3, 3]
                motion_hist.append(float(np.linalg.norm(cam_c - cam_p)))
                if len(motion_hist) > 20:
                    motion_hist.pop(0)

                # Reset KF anchor
                last_kf_Twb     = _T_wb_from_T_cw(T_cur, T_cam_imu)
                last_kf_v       = v_cur.copy()
                # Clamp velocity: hand-held camera should not exceed 5 m/s.
                # Prevents unbounded IMU-propagation drift when visual tracking
                # is lost for several consecutive keyframes.
                _vm = float(np.linalg.norm(last_kf_v))
                if _vm > 5.0:
                    last_kf_v *= 5.0 / _vm
                kf_imu          = []
                kf_pts_dict     = {int(tid): pt
                                   for tid, pt in zip(ids_t, curr_pts_t)}
                frames_since_kf = 0

            else:
                # d. Non-keyframe: IMU-propagated pose only
                T_cur = T_cw_prop

            # e. Record current pose
            trajectory.append((ts, T_cur))
            T_prev = T_cur

            # f. Log every 20 frames
            if idx % 20 == 0:
                v_mag = float(np.linalg.norm(last_kf_v))
                print(f"[Frame {idx:4d}]  tracked={n_tracked:4d}  "
                      f"map={len(landmark_map):5d}  "
                      f"cost={sw_cost:.4f}  |v|={v_mag:.3f} m/s")

    except Exception:
        print("\n[ERROR] Pipeline crashed — saving partial results.")
        traceback.print_exc()

    # ── 4. Save results ────────────────────────────────────────────────────────
    elapsed  = time.time() - t_wall_start
    n_frames = len(trajectory)

    print(f"\nTotal runtime    : {elapsed:.1f} s")
    print(f"Mean ms/frame    : {1000.0 * elapsed / max(1, n_frames):.1f} ms")
    print(f"Keyframes        : {n_keyframes}")
    print(f"Tracking failures: {n_failures}")
    print(f"VIO init frame   : {init_frame_idx}")

    if not trajectory:
        print("[WARN] No trajectory to save.")
        return

    timestamps_s = [ts for ts, _  in trajectory]
    poses_cw     = [T  for _,  T  in trajectory]

    # TUM format expects world-from-camera; invert our camera-from-world poses
    poses_wc = [np.linalg.inv(T) for T in poses_cw]
    save_tum_trajectory(poses_wc, timestamps_s, str(traj_path))

    # c. Top-down trajectory plot (blue = VIO, orange = VO reference)
    positions = np.array([T[:3, 3] for T in poses_wc])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(positions[:, 0], positions[:, 2],
            linewidth=1.0, color="steelblue", label="VIO estimate")

    vo_traj = Path("results/trajectories") / f"vo_{seq_name}.txt"
    if vo_traj.exists():
        try:
            vo_pos = []
            with open(vo_traj) as fv:
                for line in fv:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = line.split()
                    if len(p) >= 4:
                        vo_pos.append([float(p[1]), float(p[2]), float(p[3])])
            if vo_pos:
                vo_arr = np.array(vo_pos)
                ax.plot(vo_arr[:, 0], vo_arr[:, 2],
                        linewidth=0.8, color="darkorange",
                        alpha=0.7, label="VO (reference)")
        except Exception:
            pass

    ax.scatter(positions[0,  0], positions[0,  2],
               c="green", s=60, zorder=5, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 2],
               c="red",   s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z — forward (m)")
    ax.set_title(f"VIO trajectory — {seq_name}")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_plot_dir / f"vio_{seq_name}.png"
    plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Trajectory saved → {traj_path}")
    print(f"Plot saved       → {plot_path}")


if __name__ == "__main__":
    main()
