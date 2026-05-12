"""PnP + RANSAC pose estimation from 2-D/3-D correspondences."""

import cv2
import numpy as np
from pathlib import Path


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rvec/tvec to a 4x4 SE(3) matrix."""
    R, _ = cv2.Rodrigues(rvec.ravel())
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T


def _T_to_rvec_tvec(T: np.ndarray):
    """Extract rvec and tvec from a 4x4 SE(3) matrix."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec.astype(np.float64), tvec.astype(np.float64)


class PnPSolver:
    """Estimates camera pose from 3-D/2-D correspondences via PnP + RANSAC.

    Sign convention
    ---------------
    The returned T_world_cam is a **camera-from-world** transform:

        X_cam = T_world_cam @ X_world

    i.e. the 3×3 rotation block R and translation t satisfy

        X_cam = R X_world + t

    This matches the OpenCV convention returned by solvePnPRansac.
    """

    def __init__(self,
                 K: np.ndarray,
                 ransac_thresh: float = 4.0,
                 confidence: float = 0.999,
                 min_inliers: int = 12):
        self.K           = K.astype(np.float64)
        self.ransac_thresh = ransac_thresh
        self.confidence  = confidence
        self.min_inliers = min_inliers

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self,
              pts3d: np.ndarray,
              pts2d: np.ndarray,
              initial_pose: np.ndarray = None):
        """Estimate camera pose with PnP + RANSAC.

        Uses SOLVEPNP_ITERATIVE so that an initial pose guess can be
        supplied to warm-start the solver (helpful in the sliding-window
        backend where the previous frame's pose is a good prior).

        Parameters
        ----------
        pts3d : (N, 3) float64
            3-D world points.
        pts2d : (N, 2) float32
            Corresponding observed 2-D image points.
        initial_pose : (4, 4) SE(3) or None
            If given, its R and t seed the RANSAC solver.

        Returns
        -------
        T_world_cam : (4, 4) float64
            Camera-from-world transform (X_cam = T_world_cam @ X_world).
        inlier_mask : (N,) bool
            True for the RANSAC inliers.

        Raises
        ------
        RuntimeError
            If fewer than min_inliers survive RANSAC.
        """
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        pts2d = np.asarray(pts2d, dtype=np.float32).reshape(-1, 2)

        use_guess = initial_pose is not None
        if use_guess:
            init_rvec, init_tvec = _T_to_rvec_tvec(initial_pose)
        else:
            init_rvec = np.zeros((3, 1), dtype=np.float64)
            init_tvec = np.zeros((3, 1), dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d,
            self.K, None,                       # no distortion (pre-undistorted)
            rvec=init_rvec, tvec=init_tvec,
            useExtrinsicGuess=use_guess,
            iterationsCount=200,
            reprojectionError=self.ransac_thresh,
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok or inliers is None or len(inliers) < self.min_inliers:
            n = len(inliers) if (inliers is not None) else 0
            raise RuntimeError(
                f"PnP RANSAC failed: only {n} inliers "
                f"(need {self.min_inliers}). "
                "Check 3-D/2-D correspondence quality."
            )

        inlier_mask = np.zeros(len(pts3d), dtype=bool)
        inlier_mask[inliers.ravel()] = True

        T_world_cam = _rvec_tvec_to_T(rvec, tvec)
        return T_world_cam, inlier_mask

    def refine(self,
               pts3d: np.ndarray,
               pts2d: np.ndarray,
               T: np.ndarray) -> np.ndarray:
        """Non-linearly refine a PnP pose with Levenberg-Marquardt.

        Runs cv2.solvePnPRefineLM on the full inlier set, tightening the
        reprojection residuals beyond what RANSAC's coarse iteration achieves.
        Called after solve() as the final pose-polishing step in the VO loop.

        Parameters
        ----------
        pts3d : (N, 3) float64
        pts2d : (N, 2) float32
        T     : (4, 4) initial pose (e.g. from solve()).

        Returns
        -------
        T_refined : (4, 4) float64 refined pose.
        """
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        pts2d = np.asarray(pts2d, dtype=np.float32).reshape(-1, 2)

        rvec, tvec = _T_to_rvec_tvec(T)
        cv2.solvePnPRefineLM(
            pts3d, pts2d,
            self.K, None,
            rvec, tvec,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )
        return _rvec_tvec_to_T(rvec, tvec)


if __name__ == "__main__":
    # ── synthetic test ────────────────────────────────────────────────────
    # Build a known pose and project 3-D points through it, then recover the
    # pose from the 2-D projections and compare.

    np.random.seed(0)

    # Camera intrinsics (typical TUM VI)
    K = np.array([
        [190.0,   0.0, 255.0],
        [  0.0, 190.0, 256.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)

    # Ground-truth pose: small rotation + forward translation
    angle = np.deg2rad(5.0)
    R_gt  = np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [             0, 1,             0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t_gt = np.array([0.1, -0.05, 0.5])
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3]  = t_gt

    # Random 3-D points in front of the camera (world frame)
    N = 80
    pts3d = np.random.uniform(-1.0, 1.0, (N, 3))
    pts3d[:, 2] += 3.0          # push points ~3 m forward

    # Project with GT pose  (X_cam = R x + t)
    pts3d_cam = (R_gt @ pts3d.T + t_gt[:, None]).T
    uvw = (K @ pts3d_cam.T).T
    pts2d_clean = (uvw[:, :2] / uvw[:, 2:]).astype(np.float32)

    # Add mild pixel noise
    pts2d_noisy = pts2d_clean + np.random.randn(N, 2).astype(np.float32) * 0.5

    # Keep only points that project inside a 512x512 image
    valid = (
        (pts2d_noisy[:, 0] > 0) & (pts2d_noisy[:, 0] < 512) &
        (pts2d_noisy[:, 1] > 0) & (pts2d_noisy[:, 1] < 512) &
        (pts3d_cam[:, 2] > 0)
    )
    pts3d_v = pts3d[valid]
    pts2d_v = pts2d_noisy[valid]
    print(f"Synthetic points used: {valid.sum()}")

    solver = PnPSolver(K, ransac_thresh=4.0, min_inliers=8)
    T_est, mask = solver.solve(pts3d_v, pts2d_v)
    T_ref = solver.refine(pts3d_v[mask], pts2d_v[mask], T_est)

    # ── error metrics ─────────────────────────────────────────────────────
    dT   = T_ref @ np.linalg.inv(T_gt)
    R_err = np.rad2deg(np.arccos(
        np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    ))
    t_err = np.linalg.norm(dT[:3, 3])

    print(f"Inliers (RANSAC)  : {mask.sum()} / {len(pts3d_v)}")
    print(f"Rotation error    : {R_err:.4f} deg")
    print(f"Translation error : {t_err:.6f} m")
    print(f"\nT_gt  =\n{T_gt}")
    print(f"\nT_est (refined) =\n{T_ref}")
