"""Read / write TUM-format trajectory files."""

import numpy as np
from pathlib import Path
from typing import List


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [qx, qy, qz, qw]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s,
                         (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s,
                         (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                         0.25*s, (R[1,0]-R[0,1])/s])


def save_tum_trajectory(poses: List[np.ndarray],
                        timestamps: List[float],
                        path: str = "trajectory.txt") -> None:
    """Save a list of poses in TUM RGB-D / VI benchmark format.

    Each row: ``timestamp tx ty tz qx qy qz qw``

    Parameters
    ----------
    poses : list of (4, 4) float64
        **World-from-camera** SE(3) transforms (T_wc).
        T[:3, 3] is the camera centre in world coordinates.
        If your pipeline stores camera-from-world (T_cw, OpenCV convention),
        pass ``[np.linalg.inv(T) for T in poses]``.
    timestamps : list of float
        Per-frame timestamps in seconds.
    path : str
        Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = _rot_to_quat(T[:3, :3])
            f.write(
                f"{ts:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
    print(f"[OK] Trajectory saved → {path}  ({len(poses)} poses)")
