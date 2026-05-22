"""
TUM VI Dataset Loader
=====================
Handles: images, IMU data, ground truth, Kalibr calibration.
Works locally or on Google Colab (mount Drive, set DATASET_ROOT).

Usage:
    from tum_vi_loader import TUMVIDataset
    ds = TUMVIDataset("/path/to/room2")
    for frame in ds.iter_frames(max_frames=200):
        img   = frame["image"]          # undistorted grayscale (H x W, uint8)
        imu   = frame["imu_since_last"] # list of dicts {t, ax,ay,az, wx,wy,wz}
        pose  = frame["gt_pose"]        # 4x4 np.float64 or None
        stamp = frame["timestamp"]      # float seconds
"""

import os
import csv
import glob
import yaml
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterator

np.random.seed(0)

# ─────────────────────────────────────────────────────────────
# 1.  Calibration
# ─────────────────────────────────────────────────────────────

@dataclass
class CameraCalib:
    """Pinhole + radtan distortion (Kalibr format)."""
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray          # [k1, k2, p1, p2] or [k1,k2,k3,k4] equidist
    dist_model: str                  # "radtan" | "equidist"
    T_cam_imu: np.ndarray            # 4x4  (camera ← IMU)
    resolution: tuple                # (width, height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,  0,  1]], dtype=np.float64)

    @property
    def T_imu_cam(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam_imu)


@dataclass
class IMUCalib:
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    update_rate: float               # Hz


def load_kalibr_yaml(yaml_path: str) -> tuple:
    """
    Parse a Kalibr camchain-imucam.yaml.
    Returns (CameraCalib for cam0, IMUCalib).
    Falls back to TUM VI known values if file missing.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"[WARN] Calibration file not found: {yaml_path}")
        print("[WARN] Using TUM VI default calibration values.")
        return _tum_vi_default_calib()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # ── camera ──────────────────────────────────────────────
    cam = data.get("cam0", data.get("cam1", next(iter(data.values()))))
    intr = cam["intrinsics"]           # [fx, fy, cx, cy]
    dist = cam.get("distortion_coeffs", [0, 0, 0, 0])
    dist_model = cam.get("distortion_model", "radtan")
    T_cn_cnm1 = cam.get("T_cn_cnm1")  # cam←imu or cam←cam-1
    T_cam_imu_raw = cam.get("T_cam_imu", T_cn_cnm1)

    if T_cam_imu_raw is None:
        print("[WARN] T_cam_imu not found; using identity.")
        T_cam_imu = np.eye(4)
    else:
        T_cam_imu = np.array(T_cam_imu_raw, dtype=np.float64)

    res = cam.get("resolution", [512, 512])
    cam_calib = CameraCalib(
        fx=intr[0], fy=intr[1], cx=intr[2], cy=intr[3],
        dist_coeffs=np.array(dist, dtype=np.float64),
        dist_model=dist_model,
        T_cam_imu=T_cam_imu,
        resolution=tuple(res)
    )

    # ── IMU ─────────────────────────────────────────────────
    # Try a sibling imu_config.yaml (TUM VI DSO export) before the camchain entry.
    imu_cfg_path = yaml_path.parent / "imu_config.yaml"
    if imu_cfg_path.exists():
        with open(imu_cfg_path) as _f:
            imu_raw = yaml.safe_load(_f)
    else:
        imu_raw = data.get("imu0", {})

    imu_calib = IMUCalib(
        accelerometer_noise_density=imu_raw.get("accelerometer_noise_density", 2.0e-3),
        accelerometer_random_walk=imu_raw.get("accelerometer_random_walk", 3.0e-3),
        gyroscope_noise_density=imu_raw.get("gyroscope_noise_density", 1.6e-4),
        gyroscope_random_walk=imu_raw.get("gyroscope_random_walk", 1.9e-5),
        update_rate=imu_raw.get("update_rate", 200.0)
    )

    return cam_calib, imu_calib


def _tum_vi_default_calib() -> tuple:
    """
    TUM VI Room sequences — factory defaults from the benchmark paper.
    https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
    """
    cam_calib = CameraCalib(
        fx=190.97847715128717, fy=190.9733070521226,
        cx=254.93170605935475, cy=256.8974428996504,
        dist_coeffs=np.array([0.0034823894022493434,
                               0.0007150348452162257,
                              -0.0020532361418706202,
                               0.00020293673591811182]),
        dist_model="radtan",
        T_cam_imu=np.array([
            [ 0.99952904, -0.00838637,  0.02951895,  0.04589723],
            [ 0.00952318,  0.99932524, -0.03574854, -0.00148451],
            [-0.02920393,  0.03601384,  0.99891453,  0.00296617],
            [ 0.,          0.,          0.,          1.        ]
        ]),
        resolution=(512, 512)
    )
    imu_calib = IMUCalib(
        accelerometer_noise_density=2.0000e-3,
        accelerometer_random_walk=3.0000e-3,
        gyroscope_noise_density=1.6968e-4,
        gyroscope_random_walk=1.9393e-5,
        update_rate=200.0
    )
    return cam_calib, imu_calib


# ─────────────────────────────────────────────────────────────
# 2.  Raw readers
# ─────────────────────────────────────────────────────────────

def load_image_timestamps(seq_dir: str, cam: str = "cam0") -> List[tuple]:
    """
    Returns sorted list of (timestamp_ns: int, abs_path: str).
    Supports both:
      <seq>/mav0/<cam>/data/          (standard TUM VI export)
      <seq>/dso/cam0/images/          (DSO-style export)
    """
    candidates = [
        Path(seq_dir) / "mav0" / cam / "data",
        Path(seq_dir) / "dso" / cam / "images",
        Path(seq_dir) / cam / "data",
        Path(seq_dir) / "images",
    ]
    img_dir = None
    for c in candidates:
        if c.exists():
            img_dir = c
            break
    if img_dir is None:
        raise FileNotFoundError(
            f"Cannot find image directory under {seq_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    files = sorted(
        glob.glob(str(img_dir / "*.png")) +
        glob.glob(str(img_dir / "*.jpg"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    result = []
    for fp in files:
        stem = Path(fp).stem
        try:
            ts_ns = int(stem)
        except ValueError:
            ts_ns = int(stem.split("_")[0])
        result.append((ts_ns, fp))
    return result


def load_imu_data(seq_dir: str) -> np.ndarray:
    """
    Returns float64 array of shape (N, 7):
      [timestamp_ns, wx, wy, wz, ax, ay, az]
    Supports: mav0/imu0/data.csv  or  dso/imu.txt
    """
    candidates = [
        Path(seq_dir) / "mav0" / "imu0" / "data.csv",
        Path(seq_dir) / "dso" / "imu.txt",
        Path(seq_dir) / "imu0" / "data.csv",
        Path(seq_dir) / "imu.csv",
        Path(seq_dir) / "imu.txt",
    ]
    imu_file = None
    for c in candidates:
        if c.exists():
            imu_file = c
            break
    if imu_file is None:
        raise FileNotFoundError(
            f"Cannot find IMU file under {seq_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    rows = []
    with open(imu_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 7:
                rows.append([float(x) for x in parts[:7]])

    data = np.array(rows, dtype=np.float64)
    print(f"[IMU] Loaded {len(data)} measurements from {imu_file}")
    return data  # [ts_ns, wx, wy, wz, ax, ay, az]


def load_ground_truth(seq_dir: str) -> Optional[np.ndarray]:
    """
    Returns float64 array (N, 8): [ts_ns, tx, ty, tz, qx, qy, qz, qw]
    or None if not found.
    """
    candidates = [
        Path(seq_dir) / "mav0" / "mocap0" / "data.csv",
        Path(seq_dir) / "groundtruth.txt",
        Path(seq_dir) / "gt.txt",
        Path(seq_dir) / "mocap0" / "data.csv",
    ]
    gt_file = None
    for c in candidates:
        if c.exists():
            gt_file = c
            break
    if gt_file is None:
        print("[GT] No ground truth file found — will skip GT poses.")
        return None

    rows = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 8:
                rows.append([float(x) for x in parts[:8]])

    data = np.array(rows, dtype=np.float64)
    print(f"[GT]  Loaded {len(data)} poses from {gt_file}")
    return data  # [ts_ns, tx, ty, tz, qx, qy, qz, qw]


# ─────────────────────────────────────────────────────────────
# 3.  Geometry helpers
# ─────────────────────────────────────────────────────────────

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """qx, qy, qz, qw  →  3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)]
    ], dtype=np.float64)


def pose_to_matrix(row: np.ndarray) -> np.ndarray:
    """[ts, tx, ty, tz, qx, qy, qz, qw] → 4x4 SE(3) matrix."""
    tx, ty, tz = row[1], row[2], row[3]
    R = quat_to_rotation_matrix(row[4:8])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def undistort_image(img: np.ndarray, calib: CameraCalib) -> np.ndarray:
    """Undistort using radtan or equidistant (Kannala-Brandt) model."""
    if calib.dist_model in ("equidist", "equidistant"):
        # TUM VI uses Kalibr's equidistant model → cv2.fisheye (4 coefficients)
        d = calib.dist_coeffs[:4].reshape(1, 4).astype(np.float64)
        return cv2.fisheye.undistortImage(img, calib.K, d, Knew=calib.K)
    else:
        return cv2.undistort(img, calib.K, calib.dist_coeffs)


# ─────────────────────────────────────────────────────────────
# 4.  GT interpolation
# ─────────────────────────────────────────────────────────────

def interpolate_gt_pose(gt: np.ndarray, ts_ns: int) -> Optional[np.ndarray]:
    """
    Linear interpolation of ground truth at timestamp ts_ns.
    Returns 4x4 SE(3) matrix or None if outside GT range.
    """
    if gt is None or len(gt) == 0:
        return None
    ts = gt[:, 0]
    if ts_ns < ts[0] or ts_ns > ts[-1]:
        return None
    idx = np.searchsorted(ts, ts_ns)
    if idx == 0:
        return pose_to_matrix(gt[0])
    if idx >= len(ts):
        return pose_to_matrix(gt[-1])

    t0, t1 = ts[idx-1], ts[idx]
    alpha = (ts_ns - t0) / (t1 - t0 + 1e-12)

    # interpolate translation
    p0 = gt[idx-1, 1:4]
    p1 = gt[idx,   1:4]
    p  = p0 + alpha * (p1 - p0)

    # slerp quaternion [qx, qy, qz, qw]
    q0 = gt[idx-1, 4:8]
    q1 = gt[idx,   4:8]
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
    else:
        theta0 = np.arccos(dot)
        theta  = alpha * theta0
        q = (np.sin(theta0 - theta) * q0 + np.sin(theta) * q1) / np.sin(theta0)
    q /= np.linalg.norm(q)

    T = np.eye(4)
    T[:3, :3] = quat_to_rotation_matrix(q)
    T[:3, 3]  = p
    return T


# ─────────────────────────────────────────────────────────────
# 5.  Main dataset class
# ─────────────────────────────────────────────────────────────

class TUMVIDataset:
    """
    Main interface for TUM VI sequences.

    Parameters
    ----------
    seq_dir : str
        Root of a single sequence, e.g. "/data/room2"
    calib_yaml : str, optional
        Path to Kalibr camchain-imucam.yaml.
        If None, looks for it inside seq_dir, then falls back to defaults.
    cam : str
        Which camera to use ("cam0" = left, "cam1" = right).
    """

    def __init__(self, seq_dir: str,
                 calib_yaml: Optional[str] = None,
                 cam: str = "cam0"):
        self.seq_dir = str(seq_dir)
        self.cam     = cam

        # ── calibration ──────────────────────────────────────
        if calib_yaml is None:
            guesses = [
                Path(seq_dir) / "camchain-imucam.yaml",
                Path(seq_dir) / "calib" / "camchain-imucam.yaml",
                Path(seq_dir) / "mav0" / "camchain-imucam.yaml",
                Path(seq_dir) / "dso" / "camchain.yaml",   # TUM VI DSO export
            ]
            calib_yaml = next((str(g) for g in guesses if g.exists()), "NOTFOUND")

        self.cam_calib, self.imu_calib = load_kalibr_yaml(calib_yaml)

        # ── data ──────────────────────────────────────────────
        self.image_list = load_image_timestamps(seq_dir, cam)
        self.imu_data   = load_imu_data(seq_dir)          # (N,7)
        self.gt_data    = load_ground_truth(seq_dir)      # (M,8) or None

        print(f"\n[Dataset] {Path(seq_dir).name}")
        print(f"  Images : {len(self.image_list)}")
        print(f"  IMU    : {len(self.imu_data)}")
        print(f"  GT     : {len(self.gt_data) if self.gt_data is not None else 'N/A'}")
        print(f"  K      :\n{self.cam_calib.K}")

    # ── IMU windowing ─────────────────────────────────────────

    def get_imu_between(self, t_start_ns: int, t_end_ns: int) -> List[Dict]:
        """IMU measurements in (t_start, t_end] as list of dicts."""
        ts = self.imu_data[:, 0]
        mask = (ts > t_start_ns) & (ts <= t_end_ns)
        rows = self.imu_data[mask]
        return [{"t": r[0] * 1e-9,
                 "wx": r[1], "wy": r[2], "wz": r[3],
                 "ax": r[4], "ay": r[5], "az": r[6]} for r in rows]

    # ── frame iterator ────────────────────────────────────────

    def iter_frames(self, max_frames: Optional[int] = None,
                    undistort: bool = True) -> Iterator[Dict]:
        """
        Yields dicts with keys:
          index, timestamp, image, imu_since_last, gt_pose
        """
        prev_ts_ns = None
        total = len(self.image_list) if max_frames is None else min(max_frames, len(self.image_list))

        for i, (ts_ns, img_path) in enumerate(self.image_list[:total]):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read {img_path}. Skipping.")
                continue

            if undistort:
                img = undistort_image(img, self.cam_calib)

            imu_segment = []
            if prev_ts_ns is not None:
                imu_segment = self.get_imu_between(prev_ts_ns, ts_ns)

            gt_pose = interpolate_gt_pose(self.gt_data, ts_ns)

            yield {
                "index":          i,
                "timestamp":      ts_ns * 1e-9,      # seconds
                "timestamp_ns":   ts_ns,
                "image":          img,
                "imu_since_last": imu_segment,
                "gt_pose":        gt_pose,
                "path":           img_path,
            }
            prev_ts_ns = ts_ns

    # ── convenience ───────────────────────────────────────────

    def get_frame(self, index: int, undistort: bool = True) -> Dict:
        """Load a single frame by index."""
        ts_ns, img_path = self.image_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if undistort:
            img = undistort_image(img, self.cam_calib)
        gt_pose = interpolate_gt_pose(self.gt_data, ts_ns)
        imu = []
        if index > 0:
            prev_ts = self.image_list[index - 1][0]
            imu = self.get_imu_between(prev_ts, ts_ns)
        return {"index": index, "timestamp": ts_ns * 1e-9,
                "image": img, "imu_since_last": imu, "gt_pose": gt_pose}


# ─────────────────────────────────────────────────────────────
# 6.  Sanity check & visualisation
# ─────────────────────────────────────────────────────────────

def run_sanity_check(dataset: TUMVIDataset, n_frames: int = 5):
    """
    Loads first n_frames, prints stats, saves a verification strip.
    Call this after constructing TUMVIDataset to confirm everything works.
    """
    import matplotlib.pyplot as plt

    print("\n=== Sanity Check ===")
    frames = []
    for frame in dataset.iter_frames(max_frames=n_frames):
        n_imu = len(frame["imu_since_last"])
        has_gt = frame["gt_pose"] is not None
        print(f"  Frame {frame['index']:4d}  t={frame['timestamp']:.3f}s  "
              f"IMU={n_imu:3d}  GT={'yes' if has_gt else 'no '}  "
              f"img={frame['image'].shape}")
        frames.append(frame)

    if not frames:
        print("[ERROR] No frames loaded. Check dataset path.")
        return

    # ── figure: image strip + IMU acc plot ───────────────────
    fig, axes = plt.subplots(2, len(frames),
                             figsize=(3 * len(frames), 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    if len(frames) == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, frame in enumerate(frames):
        axes[0, i].imshow(frame["image"], cmap="gray")
        axes[0, i].set_title(f"t={frame['timestamp']:.2f}s", fontsize=9)
        axes[0, i].axis("off")

        imu = frame["imu_since_last"]
        if imu:
            acc_norms = [np.sqrt(m["ax"]**2 + m["ay"]**2 + m["az"]**2) for m in imu]
            axes[1, i].plot(acc_norms, linewidth=1, color="steelblue")
            axes[1, i].set_ylim(0, 20)
            axes[1, i].set_ylabel("|acc| m/s²", fontsize=7)
        else:
            axes[1, i].text(0.5, 0.5, "no IMU", ha="center", va="center",
                            transform=axes[1, i].transAxes, fontsize=8)
        axes[1, i].tick_params(labelsize=7)

    plt.suptitle(f"TUM VI — {Path(dataset.seq_dir).name} — first {len(frames)} frames",
                 fontsize=11)
    plt.tight_layout()

    out = Path("sanity_check.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\n[OK] Sanity check figure saved to {out.resolve()}")

    # ── GT trajectory preview (if available) ──────────────────
    if dataset.gt_data is not None:
        gt = dataset.gt_data
        fig2, ax = plt.subplots(figsize=(5, 5))
        ax.plot(gt[:, 1], gt[:, 2], linewidth=1.2, color="darkorange")
        ax.scatter(gt[0, 1], gt[0, 2], c="green", s=60, zorder=5, label="start")
        ax.scatter(gt[-1, 1], gt[-1, 2], c="red",   s=60, zorder=5, label="end")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("Ground truth trajectory (top-down)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        gt_out = Path("gt_trajectory.png")
        plt.savefig(gt_out, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"[OK] GT trajectory saved to {gt_out.resolve()}")


# ─────────────────────────────────────────────────────────────
# 7.  TUM trajectory writer
# ─────────────────────────────────────────────────────────────

def save_tum_trajectory(poses: List[np.ndarray],
                        timestamps: List[float],
                        path: str = "trajectory.txt"):
    """
    Save estimated SE(3) poses in TUM format:
      timestamp tx ty tz qx qy qz qw
    poses : list of 4x4 np.ndarray
    """
    def rot_to_quat(R):
        tr = R[0,0]+R[1,1]+R[2,2]
        if tr > 0:
            s = 0.5/np.sqrt(tr+1.0)
            return np.array([
                (R[2,1]-R[1,2])*s,
                (R[0,2]-R[2,0])*s,
                (R[1,0]-R[0,1])*s,
                0.25/s
            ])
        elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
            return np.array([0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s])
        elif R[1,1]>R[2,2]:
            s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
            return np.array([(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s])
        else:
            s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
            return np.array([(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s,(R[1,0]-R[0,1])/s])

    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = rot_to_quat(T[:3, :3])
            f.write(f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    print(f"[OK] Trajectory saved → {path}  ({len(poses)} poses)")


# ─────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Colab auto-mount ──────────────────────────────────────
    try:
        import google.colab
        IN_COLAB = True
        print("[Colab] Detected. Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        # Edit this path to match where you put the dataset on Drive:
        DATASET_ROOT = "/content/drive/MyDrive/tum_vi"
    except ImportError:
        IN_COLAB = False
        DATASET_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./room2"

    print(f"[Config] DATASET_ROOT = {DATASET_ROOT}")
    print(f"[Config] Colab        = {IN_COLAB}")

    dataset = TUMVIDataset(
        seq_dir=DATASET_ROOT,
        cam="cam0"
    )

    run_sanity_check(dataset, n_frames=5)
