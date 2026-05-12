"""Essential matrix estimation, decomposition, triangulation."""

import cv2
import numpy as np
from pathlib import Path


class EpipolarGeometry:
    """Two-view geometry: essential matrix, pose recovery, triangulation."""

    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)

    def estimate_essential(self, pts1, pts2,
                           ransac_thresh: float = 1.0,
                           confidence: float = 0.999):
        """Estimate the essential matrix E via RANSAC.

        Implements the epipolar constraint (eq. 1):

            x'^T E x = 0

        where x, x' are normalised image coordinates and E encodes the
        relative rotation and translation between two views.

        Points are normalised with K via cv2.undistortPoints before
        findEssentialMat so that ransac_thresh stays in pixel units
        (converted to normalised-plane units internally).

        Returns (E, mask) where mask is the boolean RANSAC inlier array.
        """
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)

        # Normalise pixel coords to the unit focal-length plane (K^{-1} x)
        pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, None)
        pts2_n = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, None)

        # Convert the pixel-space threshold to normalised-plane units
        focal_mean = (self.K[0, 0] + self.K[1, 1]) / 2.0
        thresh_n = ransac_thresh / focal_mean

        E, mask = cv2.findEssentialMat(
            pts1_n, pts2_n,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=confidence,
            threshold=thresh_n,
        )

        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < 5:
            raise ValueError(
                f"Only {n_inliers} RANSAC inliers — need at least 5 to "
                "recover a reliable pose. Try more feature matches or a "
                "larger ransac_thresh."
            )

        return E, mask

    def recover_pose(self, E, pts1, pts2, mask):
        """Decompose E into R and unit-translation t via the chirality check.

        Applies the decomposition (eq. 2):

            E = [t]_x R

        which yields four (R, t) candidates; cv2.recoverPose selects the
        unique solution where reconstructed points lie in front of both
        cameras (positive depth).

        Returns (R, t, pts1_inliers). t is unit-normalised (||t|| = 1).
        """
        inlier_mask = mask.ravel().astype(bool)
        pts1_in = np.asarray(pts1, dtype=np.float32)[inlier_mask]
        pts2_in = np.asarray(pts2, dtype=np.float32)[inlier_mask]

        _, R, t, recover_mask = cv2.recoverPose(
            E, pts1_in, pts2_in, cameraMatrix=self.K
        )

        chiral_mask = recover_mask.ravel().astype(bool)
        pts1_inliers = pts1_in[chiral_mask]

        return R, t, pts1_inliers

    def triangulate(self, R, t, pts1, pts2) -> np.ndarray:
        """Triangulate matched 2-D correspondences into 3-D points.

        Builds projection matrices

            P1 = K [I | 0]   P2 = K [R | t]

        and calls cv2.triangulatePoints. Converts homogeneous output to
        Euclidean (divide by w) and discards points with negative depth in
        either camera (chirality filter).

        Returns an (N, 3) float64 array of 3-D points.
        """
        P1 = self.K @ np.hstack([np.eye(3),        np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])

        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)   # (4, N)
        w = pts4d[3]
        pts3d = (pts4d[:3] / w).T.astype(np.float64)            # (N, 3)

        # Depth in camera 1 frame
        depth1 = pts3d[:, 2]
        # Depth in camera 2 frame
        depth2 = (R @ pts3d.T + t.reshape(3, 1)).T[:, 2]

        valid = (depth1 > 0) & (depth2 > 0)
        return pts3d[valid]

    def compute_scale(self, points3d: np.ndarray) -> float:
        """Return median depth as the initial scale estimate.

        Monocular VO cannot recover absolute scale from a single frame pair;
        median depth gives a numerically stable normalisation used to set the
        first baseline length before IMU or metric priors are available.
        """
        return float(np.median(points3d[:, 2]))


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    from src.utils.tum_vi_loader import TUMVIDataset
    from src.frontend.feature_detector import FeatureDetector

    ds = TUMVIDataset(str(ROOT / "data" / "room2"))
    K = ds.cam_calib.K

    frames = []
    for frame in ds.iter_frames(max_frames=2):
        frames.append(frame)

    img0, img1 = frames[0]["image"], frames[1]["image"]

    det = FeatureDetector(method="ORB", n_features=2000)
    kp0, desc0 = det.detect(img0)
    kp1, desc1 = det.detect(img1)
    pts0, pts1 = det.match(desc0, desc1, kp0, kp1)
    print(f"Feature matches: {len(pts0)}")

    epipolar = EpipolarGeometry(K)
    E, mask = epipolar.estimate_essential(pts0, pts1)

    R, t, _ = epipolar.recover_pose(E, pts0, pts1, mask)

    print(f"\nR =\n{R}")
    print(f"\nt = {t.ravel()}")

    # Use RANSAC inliers for triangulation; depth filter applied inside
    inlier_mask = mask.ravel().astype(bool)
    pts0_in = pts0[inlier_mask]
    pts1_in = pts1[inlier_mask]
    points3d = epipolar.triangulate(R, t, pts0_in, pts1_in)

    scale = epipolar.compute_scale(points3d)
    print(f"\n3-D points : {len(points3d)}")
    print(f"Scale (median depth) : {scale:.4f} m")

    out_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points3d[:, 0], points3d[:, 2], s=1, alpha=0.5, color="steelblue")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z — depth (m)")
    ax.set_title("Initial point cloud — top-down view")
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = out_dir / "init_pointcloud.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")
