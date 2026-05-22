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
ALL_SEQS        = ["room2", "corridor3", "outdoors5"]
PARTIAL_GT_SEQS = {"corridor3", "outdoors5"}   # GT does not cover the full run

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

    # seq can be a short name ("room2") or an absolute path on any drive.
    # Use only the final directory name for output file naming.
    seq_name = Path(seq).name

    traj_path = ROOT / "results" / "trajectories" / f"{method}_{seq_name}.txt"
    if not traj_path.exists():
        print(f"  [SKIP] trajectory not found: {traj_path}")
        return None

    # Resolve the dataset directory: absolute path takes precedence over
    # the default ROOT/data/<name> location.
    seq_path = Path(seq)
    if seq_path.is_absolute():
        data_dir = seq_path
    else:
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
    cfg     = _load_cfg(seq_name)
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

    # ── End-to-end drift (partial-GT sequences only) ───────────────────────
    # Align the full estimated trajectory with the same transform, then measure
    # how far the last estimated position is from the last available GT position.
    end_drift_m = None
    if seq_name in PARTIAL_GT_SEQS:
        ones_all    = np.ones((len(pos_est), 1), dtype=np.float64)
        pos_al_all  = (T_align @ np.hstack([pos_est, ones_all]).T).T[:, :3]
        end_pos_est = pos_al_all[-1]
        end_pos_gt  = pos_gt_all[-1]
        end_drift_m = float(np.linalg.norm(end_pos_est - end_pos_gt))

    # ── ATE ───────────────────────────────────────────────────────────────
    ate = compute_ate(pos_est_m, pos_gt_m, T_align, scale)

    # ── RPE ───────────────────────────────────────────────────────────────
    # Monocular VO poses are in an arbitrary scale; apply the Umeyama scale
    # to the translation column so RPE is in the same metric units as GT.
    # Without this, dT_est[:,3] ≪ dT_gt[:,3] and the error is ~GT segment
    # length rather than the actual trajectory drift.
    def _scale_traj(poses, s):
        out = []
        for T in poses:
            T_s = T.copy()
            T_s[:3, 3] *= s
            out.append(T_s)
        return out

    poses_est_m = _scale_traj([poses_est_full[i] for i in idx_est], scale)
    poses_gt_m  = [poses_gt_all[j] for j in idx_gt]
    rpe         = compute_rpe(poses_est_m, poses_gt_m, segment_len_m=rpe_seg)
    rpe_vals    = _rpe_errors(poses_est_m, poses_gt_m, rpe_seg)

    # ── CSV ───────────────────────────────────────────────────────────────
    tbl_dir = ROOT / "results" / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tbl_dir / f"{method}_{seq_name}_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_s", "ate_m"])
        for ts, err in zip(ts_m, per_frame_err):
            w.writerow([f"{ts:.9f}", f"{err:.6f}"])
        if end_drift_m is not None:
            w.writerow(["end_drift_m", f"{end_drift_m:.6f}"])
    print(f"  CSV  → {csv_path}")

    # ── ATE plot ──────────────────────────────────────────────────────────
    plot_dir = ROOT / "results" / "plots" / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(_STYLE) if _STYLE in plt.style.available else None
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150,
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        f"{seq_name}  |  {method.upper()}  |  ATE RMSE = {ate['rmse']:.4f} m",
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
    ate_png = plot_dir / f"{method}_{seq_name}_ate.png"
    plt.savefig(str(ate_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ATE plot → {ate_png}")

    # ── RPE histogram ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.suptitle(
        f"{seq_name}  |  {method.upper()}  |  RPE mean = {rpe['mean']:.4f} m  "
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
    rpe_png = plot_dir / f"{method}_{seq_name}_rpe.png"
    plt.savefig(str(rpe_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  RPE plot → {rpe_png}")

    return {
        "seq":          seq_name,
        "method":       method,
        "ate_rmse":     ate["rmse"],
        "rpe_mean":     rpe["mean"],
        "n_matched":    n_matched,      # frames with a GT timestamp match
        "n_no_gt":      n_failures,     # frames outside GT coverage (not VO errors)
        "end_drift_m":  end_drift_m,    # None for full-GT sequences
    }


# ── summary table ──────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results to display.")
        return

    hdr = (f"{'Seq':<12} | {'Method':<6} | {'ATE rmse (m)':>12} | "
           f"{'RPE mean (m)':>12} | {'GT frames':>9} | {'No GT':>5} | "
           f"{'end_drift_m':>11}")
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        drift = r.get("end_drift_m")
        drift_str = f"{drift:>11.4f}" if drift is not None else f"{'N/A':>11}"
        print(
            f"{r['seq']:<12} | {r['method']:<6} | "
            f"{r['ate_rmse']:>12.4f} | {r['rpe_mean']:>12.4f} | "
            f"{r['n_matched']:>9} | {r['n_no_gt']:>5} | {drift_str}"
        )
    print(sep + "\n")


def _print_comparison_table(seqs: list, vo_res: dict, vio_res: dict) -> None:
    """Extended side-by-side VO vs VIO table (used with --method all)."""
    hdr = (f"{'Seq':<12} | {'VO ATE (m)':>10} | {'VIO ATE (m)':>11} | "
           f"{'Improvement (%)':>15} | {'end_drift_m':>11}")
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for seq in seqs:
        seq_name = Path(seq).name
        vo_r  = vo_res.get(seq_name)
        vio_r = vio_res.get(seq_name)

        vo_ate_str  = f"{vo_r['ate_rmse']:>10.3f}"  if vo_r  else f"{'N/A':>10}"
        vio_ate_str = f"{vio_r['ate_rmse']:>11.3f}" if vio_r else f"{'N/A':>11}"

        if vo_r and vio_r and vo_r["ate_rmse"] > 0:
            impr = (vo_r["ate_rmse"] - vio_r["ate_rmse"]) / vo_r["ate_rmse"] * 100.0
            impr_str = f"{impr:>+.1f}%"
        else:
            impr_str = "N/A"

        drift = vio_r.get("end_drift_m") if vio_r else None
        drift_str = f"{drift:>11.3f}" if drift is not None else f"{'N/A':>11}"

        print(f"{seq_name:<12} | {vo_ate_str} | {vio_ate_str} | "
              f"{impr_str:>15} | {drift_str}")
    print(sep + "\n")


# ── ablation helpers ───────────────────────────────────────────────────────────

def _ate_rmse_for_file(traj_file: str, seq: str,
                        allow_scale: bool = True) -> float | None:
    """Compute ATE RMSE for *traj_file* against *seq* ground truth.

    Returns None when the file is missing, the dataset has no GT, or fewer
    than 5 timestamps can be matched.
    """
    if not Path(traj_file).exists():
        return None

    seq_path = Path(seq)
    data_dir = seq_path if seq_path.is_absolute() else ROOT / "data" / seq
    if not data_dir.exists():
        return None

    try:
        ts_est, pos_est = load_tum_trajectory(traj_file)
        ds = TUMVIDataset(str(data_dir))
        if ds.gt_data is None or len(ds.gt_data) == 0:
            return None

        ts_gt  = ds.gt_data[:, 0] * 1e-9
        pos_gt = ds.gt_data[:, 1:4]

        idx_est, idx_gt, _ = match_timestamps(ts_est, ts_gt, max_diff=0.02)
        if len(idx_est) < 5:
            return None

        T_align, scale = umeyama_alignment(pos_est[idx_est], pos_gt[idx_gt],
                                           allow_scale=allow_scale)
        ate = compute_ate(pos_est[idx_est], pos_gt[idx_gt], T_align, scale)
        return float(ate["rmse"])
    except Exception:
        return None


def run_ablation(seq: str, traj_dir: str) -> None:
    """Print an ATE ablation table for *seq* using pre-saved trajectory variants.

    Variants loaded from *traj_dir*:
      vio_{seq}.txt           — full VIO (visual + IMU + bias)
      vio_{seq}_nobias.txt    — VIO without bias estimation
      vio_{seq}_noimupre.txt  — VO-only (falls back to vo_{seq}.txt)
    """
    seq_name = Path(seq).name
    td       = Path(traj_dir)

    noimupre_path = td / f"vio_{seq_name}_noimupre.txt"
    vo_fallback   = td / f"vo_{seq_name}.txt"
    noimupre      = str(noimupre_path) if noimupre_path.exists() else str(vo_fallback)

    variants = [
        ("full VIO",         str(td / f"vio_{seq_name}.txt"),         False),
        ("VIO no-bias",      str(td / f"vio_{seq_name}_nobias.txt"),   False),
        ("VO-only (no IMU)", noimupre,                                  True),
    ]

    hdr = f"  {'Variant':<22} | {'ATE RMSE (m)':>12}"
    sep = "-" * len(hdr)
    print(f"\n[Ablation]  seq={seq_name}")
    print(sep)
    print(hdr)
    print(sep)
    for label, path, allow_scale in variants:
        ate = _ate_rmse_for_file(path, seq, allow_scale=allow_scale)
        ate_str = f"{ate:>12.4f}" if ate is not None else f"{'N/A':>12}"
        print(f"  {label:<22} | {ate_str}")
    print(sep + "\n")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate VO/VIO trajectory against TUM VI ground truth."
    )
    ap.add_argument("--seq",    default="room2",
                    help="Sequence name or 'all'.")
    ap.add_argument("--method", default="vo",
                    choices=["vo", "vio", "all"],
                    help="Which trajectory to load: vo, vio, or all (default: vo).")
    ap.add_argument("--ablation", action="store_true",
                    help="Run ATE ablation study across trajectory variants.")
    args = ap.parse_args()

    seqs     = ALL_SEQS if args.seq == "all" else [args.seq]
    traj_dir = str(ROOT / "results" / "trajectories")

    if args.ablation:
        for seq in seqs:
            run_ablation(seq, traj_dir)
        return

    if args.method == "all":
        # Run both methods; show side-by-side comparison table.
        vo_res: dict  = {}
        vio_res: dict = {}
        for seq in seqs:
            seq_name = Path(seq).name
            print(f"\n[Evaluating]  seq={seq}  method=vo")
            r = evaluate_sequence(seq, "vo")
            if r is not None:
                vo_res[seq_name] = r
            print(f"\n[Evaluating]  seq={seq}  method=vio")
            r = evaluate_sequence(seq, "vio")
            if r is not None:
                vio_res[seq_name] = r
        _print_comparison_table(seqs, vo_res, vio_res)
    else:
        # Backward-compatible single-method path.
        results = []
        for seq in seqs:
            print(f"\n[Evaluating]  seq={seq}  method={args.method}")
            row = evaluate_sequence(seq, args.method)
            if row is not None:
                results.append(row)
        _print_table(results)


if __name__ == "__main__":
    main()
