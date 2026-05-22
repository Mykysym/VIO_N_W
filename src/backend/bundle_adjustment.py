"""Motion-only bundle adjustment via Levenberg-Marquardt."""

import cv2
import numpy as np
from scipy.optimize import least_squares


def project(K: np.ndarray,
            R: np.ndarray,
            t: np.ndarray,
            pts3d: np.ndarray) -> np.ndarray:
    """Project (N,3) world points through K[R|t] into pixel coordinates.

    Used by the BA residual function and for external reprojection checks.
    Points behind the camera (z <= 0) have their depth clamped to 1e-6 so
    the Jacobian stays finite; they will produce large residuals and are
    effectively down-weighted by the Huber loss.

    Returns (N, 2) float64 pixel coordinates.
    """
    pts_cam = (R @ pts3d.T + t.reshape(3, 1)).T  # (N, 3)
    z = np.maximum(pts_cam[:, 2], 1e-6)
    uvw = (K @ pts_cam.T).T                      # (N, 3)
    return uvw[:, :2] / uvw[:, 2:3]


def _pose_vec_to_Rt(pose_vec: np.ndarray):
    """Decode a 6-vector [angle_axis(3) | t(3)] → (R 3×3, t 3×1)."""
    R, _ = cv2.Rodrigues(pose_vec[:3].reshape(3, 1))
    t    = pose_vec[3:].reshape(3, 1)
    return R, t


def _T_to_pose_vec(T: np.ndarray) -> np.ndarray:
    """Encode a 4×4 SE(3) matrix as a 6-vector [angle_axis | t]."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([rvec.ravel(), T[:3, 3]])


class MotionOnlyBA:
    """Motion-only bundle adjustment: optimise a single camera pose.

    Holds 3-D structure fixed and minimises the total reprojection error
    with respect to the 6-DoF pose parameterised as [angle_axis | t].
    Suitable as a final pose-polishing step after PnP+RANSAC.
    """

    def __init__(self,
                 K: np.ndarray,
                 loss: str = "huber",
                 huber_delta: float = 1.0):
        self.K           = K.astype(np.float64)
        self.loss        = loss
        self.huber_delta = huber_delta

    # ── public API ────────────────────────────────────────────────────────────

    def optimise(self,
                 T_init: np.ndarray,
                 pts3d: np.ndarray,
                 pts2d: np.ndarray):
        """Minimise reprojection error over the 6-DoF camera pose.

        Parameterises the pose as a 6-vector [angle_axis(3) | t(3)] and
        calls scipy LM, which is well-suited to dense, small-residual
        reprojection problems. The Huber loss caps the influence of outlier
        correspondences that survived RANSAC.

        Parameters
        ----------
        T_init : (4, 4)  initial camera-from-world SE(3) pose.
        pts3d  : (N, 3) float64  fixed 3-D world points.
        pts2d  : (N, 2) float32  observed 2-D pixel coordinates.

        Returns
        -------
        T_opt : (4, 4) float64  optimised pose.
        rms   : float  final RMS reprojection error in pixels.
        """
        pts3d = np.asarray(pts3d, dtype=np.float64)
        pts2d = np.asarray(pts2d, dtype=np.float64)

        pose0 = _T_to_pose_vec(T_init)

        if self.loss == "huber":
            method = "trf"
            loss   = "huber"
            f_scale = self.huber_delta
        else:
            method = "lm"
            loss   = "linear"
            f_scale = 1.0

        result = least_squares(
            self.residuals,
            pose0,
            args=(pts3d, pts2d),
            method=method,
            loss=loss,
            f_scale=f_scale,
            max_nfev=500,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        R_opt, t_opt = _pose_vec_to_Rt(result.x)
        T_opt = np.eye(4, dtype=np.float64)
        T_opt[:3, :3] = R_opt
        T_opt[:3, 3]  = t_opt.ravel()

        raw_res = self.residuals(result.x, pts3d, pts2d)
        rms = float(np.sqrt(np.mean(raw_res ** 2)))

        return T_opt, rms

    def residuals(self,
                  pose_vec: np.ndarray,
                  pts3d: np.ndarray,
                  pts2d: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals for all correspondences.

        Returns a flat (2N,) float64 array of (u_proj-u_obs, v_proj-v_obs)
        interleaved per point.  This is the vector that least_squares
        minimises in the L2 (or robustified) sense.
        """
        R, t = _pose_vec_to_Rt(pose_vec)
        proj  = project(self.K, R, t, pts3d)        # (N, 2)
        diff  = (proj - pts2d).ravel()               # (2N,)
        return diff


if __name__ == "__main__":
    np.random.seed(42)

    # ── camera ───────────────────────────────────────────────────────────
    K = np.array([
        [190.0,   0.0, 255.0],
        [  0.0, 190.0, 256.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)

    # ── ground-truth pose ────────────────────────────────────────────────
    angle = np.deg2rad(8.0)
    R_gt  = np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [             0, 1,             0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t_gt = np.array([0.2, -0.1, 0.6])
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3]  = t_gt

    # ── synthetic scene ──────────────────────────────────────────────────
    N = 60
    pts3d = np.random.uniform(-1.5, 1.5, (N, 3))
    pts3d[:, 2] += 4.0          # push scene ~4 m forward

    pts2d_clean = project(K, R_gt, t_gt, pts3d)
    noise_sigma = 1.5           # px
    pts2d_noisy = pts2d_clean + np.random.randn(N, 2) * noise_sigma

    # Keep points inside a 512×512 image and in front of camera
    in_front = (R_gt @ pts3d.T + t_gt[:, None]).T[:, 2] > 0
    in_frame = (
        (pts2d_noisy[:, 0] > 0) & (pts2d_noisy[:, 0] < 512) &
        (pts2d_noisy[:, 1] > 0) & (pts2d_noisy[:, 1] < 512)
    )
    mask = in_front & in_frame
    pts3d_v = pts3d[mask]
    pts2d_v = pts2d_noisy[mask]
    print(f"Points used: {mask.sum()}")

    # ── perturbed initial pose ────────────────────────────────────────────
    # Add 2 degree rotation noise and 2 cm translation noise so the optimiser
    # has something to do.
    perturb_R, _ = cv2.Rodrigues(np.array([0.035, -0.02, 0.01]))
    T_init = T_gt.copy()
    T_init[:3, :3] = perturb_R @ R_gt
    T_init[:3, 3]  = t_gt + np.array([0.02, -0.01, 0.015])

    # ── initial error ────────────────────────────────────────────────────
    R0, t0 = T_init[:3, :3], T_init[:3, 3]
    res0 = (project(K, R0, t0, pts3d_v) - pts2d_v).ravel()
    rms0 = float(np.sqrt(np.mean(res0 ** 2)))

    # ── optimise ─────────────────────────────────────────────────────────
    ba = MotionOnlyBA(K, loss="huber", huber_delta=1.0)
    T_opt, rms_final = ba.optimise(T_init, pts3d_v, pts2d_v)

    print(f"Initial RMS reprojection error : {rms0:.4f} px")
    print(f"Final   RMS reprojection error : {rms_final:.4f} px")

    dT    = T_opt @ np.linalg.inv(T_gt)
    R_err = np.rad2deg(np.arccos(
        np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    ))
    t_err = np.linalg.norm(dT[:3, 3])
    print(f"Rotation  error vs GT : {R_err:.4f} deg")
    print(f"Translation error vs GT: {t_err:.6f} m")
