"""Compute ATE / RPE for all saved trajectories.

Usage:
    python evaluate.py --seq room2
    python evaluate.py --seq room2 --method vio   (default: vo)
    python evaluate.py --seq all
"""

import argparse
import csv
import sys
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── root and sequence list ─────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
ALL_SEQS = ["room2", "corridor3", "outdoors5"]

sys.path.insert(0, str(ROOT))

from src.utils.tum_vi_loader import TUMVIDataset, pose_to_matrix, quat_to_rotation_matrix
from src.utils.evaluation    import (
    load_tum_trajectory, match_timestamps,
    umeyama_alignment, compute_ate, compute_rpe,
)

# ── style guard (matplotlib >= 3.6 uses seaborn-v0_8-*, older uses seaborn-*) ──
_STYLE = "seaborn-v0_8-whitegrid"
try:
    plt.style.use(_STYLE)
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass    # fall back to default


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_full_poses(path: str) -> list:
    """Read a TUM trajectory file → list of (4,4) world-from-camera matrices."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
            q = np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
            R = quat_to_rotation_matrix(q)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            poses.append(T)
    return poses


def _rpe_errors(poses_est: list, poses_gt: list,
                segment_len_m: float) -> np.ndarray:
    """Return per-segment RPE translation errors as a 1-D array."""
    n = len(poses_est)
    if n < 2:
        return np.array([])

    gt_pos   = np.array([T[:3, 3] for T in poses_gt], dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])

    errors = []
    for i in range(n):
        target = cum_dist[i] + segment_len_m
        j      = int(np.argmin(np.abs(cum_dist - target)))
        if j <= i or abs(cum_dist[j] - target) > segment_len_m * 0.25:
            continue
        dT_est = np.linalg.inv(poses_est[i]) @ poses_est[j]
        dT_gt  = np.linalg.inv(poses_gt[i])  @ poses_gt[j]
        E      = np.linalg.inv(dT_gt) @ dT_est
        errors.append(float(np.linalg.norm(E[:3, 3])))

    return np.array(errors, dtype=np.float64)


def _load_cfg(seq: str) -> dict:
    cfg_path = ROOT / "configs" / f"{seq}.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


# ── per-sequence evaluation ────────────────────────────────────────────────────

def evaluate_sequence(seq: str, method: str) -> dict | None:
    """Evaluate one (seq, method) pair.  Returns summary dict or None on error."""

    traj_path = ROOT / "results" / "trajectories" / f"{method}_{seq}.txt"
    if not traj_path.exists():
        print(f"  [SKIP] trajectory not found: {traj_path}")
        return None

    data_dir = ROOT / "data" / seq
    if not data_dir.exists():
        print(f"  [SKIP] dataset directory not found: {data_dir}")
        return None

    # ── load estimated ────────────────────────────────────────────────────
    ts_est,  pos_est       = load_tum_trajectory(str(traj_path))
    poses_est_full         = _load_full_poses(str(traj_path))

    # ── load GT ───────────────────────────────────────────────────────────
    try:
        ds = TUMVIDataset(str(data_dir))
    except Exception as exc:
        print(f"  [SKIP] could not load dataset for {seq}: {exc}")
        return None

    if ds.gt_data is None or len(ds.gt_data) == 0:
        print(f"  [SKIP] no ground truth for {seq}")
        return None

    ts_gt        = ds.gt_data[:, 0] * 1e-9          # ns → s
    pos_gt_all   = ds.gt_data[:, 1:4]
    poses_gt_all = [pose_to_matrix(row) for row in ds.gt_data]

    # ── config ────────────────────────────────────────────────────────────
    cfg     = _load_cfg(seq)
    ev_cfg  = cfg.get("eval", {})
    rpe_seg = float(ev_cfg.get("rpe_segment_len", 100.0))
    # VO always gets Sim(3) unless config says se3; VIO always SE(3)
    align_str  = ev_cfg.get("align", "sim3").lower()
    allow_scale = (method == "vo") and (align_str == "sim3")

    # ── timestamp matching ─────────────────────────────────────────────────
    idx_est, idx_gt, diffs = match_timestamps(ts_est, ts_gt, max_diff=0.02)
    n_matched  = len(idx_est)
    n_failures = len(ts_est) - n_matched

    if n_matched < 5:
        print(f"  [SKIP] only {n_matched} matched frames for {seq}/{method}")
        return None

    pos_est_m = pos_est[idx_est]
    pos_gt_m  = pos_gt_all[idx_gt]
    ts_m      = ts_est[idx_est]

    # ── alignment ─────────────────────────────────────────────────────────
    T_align, scale = umeyama_alignment(pos_est_m, pos_gt_m, allow_scale=allow_scale)

    # Aligned estimated positions
    ones        = np.ones((n_matched, 1), dtype=np.float64)
    pos_aligned = (T_align @ np.hstack([pos_est_m, ones]).T).T[:, :3]
    per_frame_err = np.linalg.norm(pos_aligned - pos_gt_m, axis=1)

    # ── ATE ───────────────────────────────────────────────────────────────
    ate = compute_ate(pos_est_m, pos_gt_m, T_align, scale)

    # ── RPE ───────────────────────────────────────────────────────────────
    poses_est_m = [poses_est_full[i] for i in idx_est]
    poses_gt_m  = [poses_gt_all[j]   for j in idx_gt]
    rpe         = compute_rpe(poses_est_m, poses_gt_m, segment_len_m=rpe_seg)
    rpe_vals    = _rpe_errors(poses_est_m, poses_gt_m, rpe_seg)

    # ── CSV ───────────────────────────────────────────────────────────────
    tbl_dir = ROOT / "results" / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tbl_dir / f"{method}_{seq}_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_s", "ate_m"])
        for ts, err in zip(ts_m, per_frame_err):
            w.writerow([f"{ts:.9f}", f"{err:.6f}"])
    print(f"  CSV  → {csv_path}")

    # ── ATE plot ──────────────────────────────────────────────────────────
    plot_dir = ROOT / "results" / "plots" / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(_STYLE) if _STYLE in plt.style.available else None
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150,
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        f"{seq}  |  {method.upper()}  |  ATE RMSE = {ate['rmse']:.4f} m",
        fontsize=13, fontweight="bold",
    )

    # top: top-down x-y
    ax0 = axes[0]
    ax0.plot(pos_gt_m[:, 0],      pos_gt_m[:, 1],      color="orange",
             linewidth=1.4, label="GT",        zorder=2)
    ax0.plot(pos_aligned[:, 0],   pos_aligned[:, 1],   color="steelblue",
             linewidth=1.0, label="Estimated", zorder=3)
    ax0.scatter(pos_gt_m[0, 0],  pos_gt_m[0, 1],  c="green", s=50,
                zorder=5, label="start")
    ax0.scatter(pos_gt_m[-1, 0], pos_gt_m[-1, 1], c="red",   s=50,
                zorder=5, label="end")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_aspect("equal")
    ax0.set_title("Top-down trajectory (x–y plane)")
    ax0.legend(fontsize=8, loc="best")

    # bottom: per-frame ATE over time
    ax1 = axes[1]
    t_rel = ts_m - ts_m[0]
    ax1.plot(t_rel, per_frame_err, color="steelblue", linewidth=0.8)
    ax1.axhline(ate["rmse"],   color="red",    linestyle="--", linewidth=1.0,
                label=f"RMSE = {ate['rmse']:.3f} m")
    ax1.axhline(ate["median"], color="orange", linestyle=":",  linewidth=1.0,
                label=f"median = {ate['median']:.3f} m")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("ATE (m)")
    ax1.set_title("Per-frame Absolute Trajectory Error")
    ax1.legend(fontsize=8)

    plt.tight_layout()
    ate_png = plot_dir / f"{method}_{seq}_ate.png"
    plt.savefig(str(ate_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ATE plot → {ate_png}")

    # ── RPE histogram ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.suptitle(
        f"{seq}  |  {method.upper()}  |  RPE mean = {rpe['mean']:.4f} m  "
        f"(segment = {rpe_seg:.0f} m)",
        fontsize=13, fontweight="bold",
    )
    if len(rpe_vals) > 0:
        ax.hist(rpe_vals, bins=min(30, max(5, len(rpe_vals) // 3)),
                color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(rpe["mean"],   color="red",    linestyle="--", linewidth=1.2,
                   label=f"mean   = {rpe['mean']:.3f} m")
        ax.axvline(rpe["median"], color="orange", linestyle=":",  linewidth=1.2,
                   label=f"median = {rpe['median']:.3f} m")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No RPE segments found\n"
                f"(trajectory shorter than {rpe_seg:.0f} m)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("RPE — translation (m)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    rpe_png = plot_dir / f"{method}_{seq}_rpe.png"
    plt.savefig(str(rpe_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  RPE plot → {rpe_png}")

    return {
        "seq":        seq,
        "method":     method,
        "ate_rmse":   ate["rmse"],
        "rpe_mean":   rpe["mean"],
        "n_frames":   n_matched,
        "n_failures": n_failures,
    }


# ── summary table ──────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results to display.")
        return

    hdr  = f"{'Seq':<12} | {'Method':<6} | {'ATE rmse (m)':>12} | {'RPE mean (m)':>12} | {'Frames':>6} | {'Failures':>8}"
    sep  = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['seq']:<12} | {r['method']:<6} | "
            f"{r['ate_rmse']:>12.4f} | {r['rpe_mean']:>12.4f} | "
            f"{r['n_frames']:>6} | {r['n_failures']:>8}"
        )
    print(sep + "\n")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate VO/VIO trajectory against TUM VI ground truth."
    )
    ap.add_argument("--seq",    default="room2",
                    help="Sequence name or 'all'.")
    ap.add_argument("--method", default="vo",
                    choices=["vo", "vio"],
                    help="Which trajectory to load (default: vo).")
    args = ap.parse_args()

    seqs = ALL_SEQS if args.seq == "all" else [args.seq]

    results = []
    for seq in seqs:
        print(f"\n[Evaluating]  seq={seq}  method={args.method}")
        row = evaluate_sequence(seq, args.method)
        if row is not None:
            results.append(row)

    _print_table(results)


if __name__ == "__main__":
    main()
