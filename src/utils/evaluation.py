"""ATE, RPE, start-end drift; Umeyama Sim(3)/SE(3) alignment."""

import numpy as np
from pathlib import Path


# ── 1. Alignment ───────────────────────────────────────────────────────────────

def umeyama_alignment(traj_est: np.ndarray,
                      traj_gt:  np.ndarray,
                      allow_scale: bool = True):
    """Closed-form Sim(3) / SE(3) alignment of two XYZ trajectory sets.

    Implements Umeyama (1991) "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns":

      eq. (6)  minimise (1/N) Σ ||q_i − (s R p_i + t)||²

      eq. (7)  Σ_PQ = UDV^T  ⟹  R = U diag(1,1,det(U)det(V)) V^T
                              s = (1/σ²_P) trace(diag(S) D)
                              t = μ_Q − s R μ_P

    Parameters
    ----------
    traj_est, traj_gt : (N, 3) float64 — XYZ positions, already time-matched.
    allow_scale : True → Sim(3) (recommended for monocular VO);
                  False → SE(3) (for metric VIO).

    Returns
    -------
    T_align : (4, 4) float64  — T_align[:3,:3] = s * R, T_align[:3,3] = t.
              Apply as: p_aligned = T_align[:3,:3] @ p_est + T_align[:3,3].
    scale   : float — recovered scale; 1.0 when allow_scale=False.
    """
    traj_est = np.asarray(traj_est, dtype=np.float64)
    traj_gt  = np.asarray(traj_gt,  dtype=np.float64)
    assert traj_est.shape == traj_gt.shape and traj_est.ndim == 2

    N = len(traj_est)
    mu_est = traj_est.mean(axis=0)
    mu_gt  = traj_gt.mean(axis=0)

    est_c = traj_est - mu_est           # (N, 3) centred
    gt_c  = traj_gt  - mu_gt

    var_est = float(np.sum(est_c ** 2) / N)

    # Cross-covariance Σ_PQ (eq. 38): (3, 3), gt on left, est on right
    sigma = (gt_c.T @ est_c) / N

    U, S, Vt = np.linalg.svd(sigma)

    # Sign correction: D = diag(1, 1, det(U)·det(V)) ensures det(R)=+1
    d_sign = float(np.sign(np.linalg.det(U) * np.linalg.det(Vt)))
    if d_sign == 0.0:
        d_sign = 1.0
    D = np.array([1.0, 1.0, d_sign])

    R = U @ np.diag(D) @ Vt

    if allow_scale and var_est > 1e-12:
        scale = float((S * D).sum() / var_est)
    else:
        scale = 1.0

    t = mu_gt - scale * R @ mu_est

    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = scale * R
    T_align[:3,  3] = t

    return T_align, scale


# ── 2. ATE ─────────────────────────────────────────────────────────────────────

def compute_ate(traj_est: np.ndarray,
                traj_gt:  np.ndarray,
                T_align:  np.ndarray,
                scale:    float = 1.0) -> dict:
    """Absolute Trajectory Error (ATE) after Umeyama alignment.

    Applies T_align to traj_est (embedding the recovered scale), then
    measures the per-frame Euclidean distance to traj_gt.  Used as the
    primary accuracy metric for monocular VO (Sim3-aligned) and VIO
    (SE3-aligned).

    Returns dict with keys: mean, rmse, median, std, max (metres).
    """
    traj_est = np.asarray(traj_est, dtype=np.float64)
    traj_gt  = np.asarray(traj_gt,  dtype=np.float64)

    # T_align already encodes s*R and t; homogeneous multiply
    ones = np.ones((len(traj_est), 1), dtype=np.float64)
    est_h = np.hstack([traj_est, ones])             # (N, 4)
    traj_aligned = (T_align @ est_h.T).T[:, :3]     # (N, 3)

    errors = np.linalg.norm(traj_aligned - traj_gt, axis=1)

    return {
        "mean":   float(errors.mean()),
        "rmse":   float(np.sqrt((errors ** 2).mean())),
        "median": float(np.median(errors)),
        "std":    float(errors.std()),
        "max":    float(errors.max()),
    }


# ── 3. RPE ─────────────────────────────────────────────────────────────────────

def compute_rpe(poses_est: list,
                poses_gt:  list,
                segment_len_m: float = 100.0) -> dict:
    """Relative Pose Error (RPE) over fixed arc-length segments.

    For each starting frame i finds the frame j whose cumulative GT arc
    distance from i is closest to segment_len_m, then measures the
    translational part of the relative pose error:

        ΔT_est = T_est_i⁻¹ T_est_j
        ΔT_gt  = T_gt_i⁻¹  T_gt_j
        error  = || trans( ΔT_gt⁻¹  ΔT_est ) ||

    Both pose lists must be world-from-camera SE(3) matrices (TUM format).
    Returns dict with keys: mean, rmse, median (metres).
    """
    n = len(poses_est)
    if n < 2:
        return {"mean": 0.0, "rmse": 0.0, "median": 0.0}

    # Cumulative arc length along GT trajectory
    gt_pos   = np.array([T[:3, 3] for T in poses_gt], dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])

    errors = []
    for i in range(n):
        target = cum_dist[i] + segment_len_m
        j      = int(np.argmin(np.abs(cum_dist - target)))
        if j <= i:
            continue
        # Accept the pair if the arc length is within ±25 % of the target
        if abs(cum_dist[j] - target) > segment_len_m * 0.25:
            continue

        dT_est = np.linalg.inv(poses_est[i]) @ poses_est[j]
        dT_gt  = np.linalg.inv(poses_gt[i])  @ poses_gt[j]
        E      = np.linalg.inv(dT_gt) @ dT_est
        errors.append(float(np.linalg.norm(E[:3, 3])))

    if not errors:
        return {"mean": 0.0, "rmse": 0.0, "median": 0.0}

    err = np.array(errors, dtype=np.float64)
    return {
        "mean":   float(err.mean()),
        "rmse":   float(np.sqrt((err ** 2).mean())),
        "median": float(np.median(err)),
    }


# ── 4. I/O helpers ─────────────────────────────────────────────────────────────

def load_tum_trajectory(path: str):
    """Read a TUM trajectory file and return positions only.

    Each non-comment line: ``timestamp tx ty tz qx qy qz qw``
    (rotation is ignored here; use _load_tum_poses for full SE(3) poses).

    Returns
    -------
    timestamps : (N,) float64 — seconds.
    poses_xyz  : (N, 3) float64 — camera centres in world frame.
    """
    timestamps = []
    positions  = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            timestamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return (np.array(timestamps, dtype=np.float64),
            np.array(positions,  dtype=np.float64))


def match_timestamps(ts_est:   np.ndarray,
                     ts_gt:    np.ndarray,
                     max_diff: float = 0.02):
    """Associate estimated and GT timestamps by nearest neighbour.

    For each timestamp in ts_est, finds the closest timestamp in ts_gt and
    discards the pair if the gap exceeds max_diff seconds.  Suitable for
    matching 20 fps camera frames to 200 Hz GT streams.

    Returns
    -------
    idx_est   : (M,) int64 — indices into ts_est.
    idx_gt    : (M,) int64 — indices into ts_gt.
    time_diffs: (M,) float64 — |ts_est[i] − ts_gt[j]| in seconds.
    """
    ts_est = np.asarray(ts_est, dtype=np.float64)
    ts_gt  = np.asarray(ts_gt,  dtype=np.float64)

    idx_est_list  = []
    idx_gt_list   = []
    diffs_list    = []

    for i, t in enumerate(ts_est):
        j    = int(np.argmin(np.abs(ts_gt - t)))
        diff = float(abs(t - ts_gt[j]))
        if diff <= max_diff:
            idx_est_list.append(i)
            idx_gt_list.append(j)
            diffs_list.append(diff)

    return (np.array(idx_est_list,  dtype=np.int64),
            np.array(idx_gt_list,   dtype=np.int64),
            np.array(diffs_list,    dtype=np.float64))


# ── __main__ ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    traj_file = ROOT / "results" / "trajectories" / "vo_room2.txt"
    if not traj_file.exists():
        print(f"[ERROR] {traj_file} not found — run run_vo.py first.")
        sys.exit(1)

    # ── load estimated trajectory ────────────────────────────────────────
    ts_est, pos_est = load_tum_trajectory(str(traj_file))

    # Load full 4×4 poses from the TUM file (needed for RPE)
    def _tum_line_to_T(parts):
        from src.utils.tum_vi_loader import quat_to_rotation_matrix
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        q = np.array([float(parts[4]), float(parts[5]),
                      float(parts[6]), float(parts[7])])
        R = quat_to_rotation_matrix(q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    poses_est_full = []
    with open(traj_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            _p = _line.split()
            if len(_p) >= 8:
                poses_est_full.append(_tum_line_to_T(_p))

    # ── load ground truth ────────────────────────────────────────────────
    from src.utils.tum_vi_loader import TUMVIDataset, pose_to_matrix

    ds = TUMVIDataset(str(ROOT / "data" / "room2"))
    if ds.gt_data is None:
        print("[ERROR] No ground truth found in data/room2.")
        sys.exit(1)

    ts_gt  = ds.gt_data[:, 0] * 1e-9        # ns → s
    pos_gt = ds.gt_data[:, 1:4]
    poses_gt_full = [pose_to_matrix(row) for row in ds.gt_data]

    # ── timestamp matching ───────────────────────────────────────────────
    idx_est, idx_gt, diffs = match_timestamps(ts_est, ts_gt)
    print(f"Matched {len(idx_est)} frames  "
          f"(mean Δt = {diffs.mean()*1e3:.2f} ms, "
          f"max Δt = {diffs.max()*1e3:.2f} ms)")

    pos_est_m = pos_est[idx_est]
    pos_gt_m  = pos_gt[idx_gt]

    # ── ATE with Sim(3) alignment ────────────────────────────────────────
    T_align, scale = umeyama_alignment(pos_est_m, pos_gt_m, allow_scale=True)
    print(f"\nUmeyama scale : {scale:.6f}")

    ate = compute_ate(pos_est_m, pos_gt_m, T_align, scale)
    print("\n── ATE (Sim3-aligned) ───────────────────────────────")
    for key, val in ate.items():
        print(f"  {key:6s}: {val:.4f} m")

    # ── RPE ──────────────────────────────────────────────────────────────
    poses_est_m = [poses_est_full[i] for i in idx_est]
    poses_gt_m  = [poses_gt_full[j]  for j in idx_gt]

    rpe = compute_rpe(poses_est_m, poses_gt_m, segment_len_m=100.0)
    print("\n── RPE (segment = 100 m) ────────────────────────────")
    for key, val in rpe.items():
        print(f"  {key:6s}: {val:.4f} m")
