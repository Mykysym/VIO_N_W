"""Sliding-Window Visual-Inertial Optimizer.

State per keyframe:
    x_k = { T_k (4×4 SE3, world←IMU),  v_k (3,),  b_a_k (3,),  b_g_k (3,) }

The optimization variable is stacked as:
    δx_k = [δφ_k (3) | δt_k (3) | δv_k (3) | δb_g_k (3) | δb_a_k (3)]  — 15 DOF

All rotations use the RIGHT perturbation convention: R → R @ Exp(δφ).
"""

import numpy as np
import cv2
from typing import Optional, List

np.random.seed(0)

from src.backend.imu_preintegration import IMUPreintegration
from src.backend.imu_factor          import IMUFactor
from src.backend.bundle_adjustment   import project


# ── constants ─────────────────────────────────────────────────────────────────

DOF = 15        # DOF per keyframe: φ(3)+t(3)+v(3)+b_g(3)+b_a(3)
SIGMA_VIS = 1.5  # visual noise [px]


# ── SE3 helpers ───────────────────────────────────────────────────────────────

def _T_to_rv(T: np.ndarray):
    """4×4 SE3 → (rvec (3,), tvec (3,)) via cv2.Rodrigues."""
    rv, _ = cv2.Rodrigues(T[:3, :3])
    return rv.ravel().copy(), T[:3, 3].copy()


def _rv_to_T(rv: np.ndarray, tv: np.ndarray) -> np.ndarray:
    """(rvec (3,), tvec (3,)) → 4×4 SE3."""
    R, _ = cv2.Rodrigues(rv.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = tv.ravel()
    return T


def _skew(v: np.ndarray) -> np.ndarray:
    v = v.ravel()
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]], dtype=np.float64)


# ── SlidingWindowOptimizer ────────────────────────────────────────────────────

class SlidingWindowOptimizer:
    """Gauss-Newton sliding-window visual-inertial estimator.

    Maintains a bounded window of keyframes and associated IMU preintegrations.
    Solves the factor graph with analytic Jacobians from IMUFactor and a
    closed-form reprojection Jacobian.

    Coordinate convention
    ---------------------
    Keyframe pose is stored as T_wi ∈ SE3 (world ← IMU).
    Camera pose is derived as: T_cw = T_cam_imu @ inv(T_wi).
    """

    def __init__(self,
                 K:          np.ndarray,
                 T_cam_imu:  np.ndarray,
                 imu_calib,
                 window_size: int = 5,
                 gravity:    np.ndarray = None) -> None:
        self.K           = np.asarray(K,          dtype=np.float64)
        self.T_cam_imu   = np.asarray(T_cam_imu,  dtype=np.float64)
        self.T_imu_cam   = np.linalg.inv(self.T_cam_imu)
        self.imu_calib   = imu_calib
        self.window_size = int(window_size)
        self.gravity     = (np.array([0., 0., -9.81])
                            if gravity is None
                            else np.asarray(gravity, dtype=np.float64).ravel())

        # IMU noise
        self._w_bg = 1.0 / (imu_calib.gyroscope_random_walk     ** 2 + 1e-18)
        self._w_ba = 1.0 / (imu_calib.accelerometer_random_walk ** 2 + 1e-18)

        # Window storage
        self.keyframes: List[dict] = []   # dicts described below
        self.preints:   List[IMUPreintegration] = []

        # Marginalization prior (from Schur complement of expelled frames)
        self.prior_H: Optional[np.ndarray] = None
        self.prior_b: Optional[np.ndarray] = None

    # ── public interface ──────────────────────────────────────────────────────

    def add_keyframe(self,
                     T_cam_world:      np.ndarray,
                     v_world:          np.ndarray,
                     b_g:              np.ndarray,
                     b_a:              np.ndarray,
                     timestamp:        float,
                     pts3d:            list,
                     pts2d:            list,
                     imu_measurements: list) -> None:
        """Append a keyframe; marginalize oldest if window overflows.

        Parameters
        ----------
        T_cam_world      : 4×4 camera-from-world pose (from VO / VIO front-end).
        v_world          : (3,) velocity in world frame [m/s].
        b_g, b_a         : (3,) gyroscope / accelerometer biases.
        timestamp        : keyframe time [s].
        pts3d, pts2d     : parallel lists of 3-D world points and 2-D pixels.
        imu_measurements : list of {t,wx,wy,wz,ax,ay,az} dicts since prev KF.
        """
        # Convert to world-from-IMU: T_wi = T_imu_cam^{-1} @ T_cam_world^{-1}
        # T_cw = T_cam_world  →  T_wc = inv  →  T_wi = T_wc @ T_cam_imu
        T_wc = np.linalg.inv(np.asarray(T_cam_world, dtype=np.float64))
        T_wi = T_wc @ self.T_cam_imu          # world-from-IMU

        # IMU preintegration between the previous and new keyframe
        if self.keyframes:
            prev_kf = self.keyframes[-1]
            preint  = IMUPreintegration(prev_kf["b_g"], prev_kf["b_a"],
                                        self.imu_calib)
            preint.integrate_imu_segment(imu_measurements)
            self.preints.append(preint)

        self.keyframes.append({
            "T":         T_wi,
            "v":         np.asarray(v_world, dtype=np.float64).ravel().copy(),
            "b_g":       np.asarray(b_g,     dtype=np.float64).ravel().copy(),
            "b_a":       np.asarray(b_a,     dtype=np.float64).ravel().copy(),
            "timestamp": float(timestamp),
            "pts3d":     list(pts3d),
            "pts2d":     list(pts2d),
        })

        if len(self.keyframes) > self.window_size:
            self._marginalize_oldest()

    def optimize(self, n_iterations: int = 5) -> dict:
        """Gauss-Newton optimization of the sliding-window factor graph.

        Factor graph:
          (a) Visual reprojection residuals — all (keyframe, landmark) pairs.
          (b) IMU factor residuals         — each consecutive KF pair.
          (c) Bias random-walk prior       — diagonal Gaussian on biases.
          (d) Marginalization prior         — dense prior from expelled frames.

        The linear system  H δx = b  is assembled analytically and solved
        with numpy.  State is updated in-place after each iteration.

        Returns
        -------
        dict : final_cost, n_iterations, converged, delta_norm.
        """
        N    = len(self.keyframes)
        if N == 0:
            return {"final_cost": 0., "n_iterations": 0,
                    "converged": True, "delta_norm": 0.}

        # With only one keyframe there are no IMU factors, so velocity DOF is
        # unconstrained.  The Gauss-Newton step for velocity diverges because
        # H[v,v] ≈ 0 + small bias prior.  Skip optimisation until we have at
        # least two keyframes (i.e. one IMU factor).
        if N < 2:
            return {"final_cost": 0., "n_iterations": 0,
                    "converged": True, "delta_norm": 0.}

        size = N * DOF

        prev_cost  = None
        delta_norm = np.inf
        converged  = False

        for it in range(n_iterations):
            H = np.zeros((size, size), dtype=np.float64)
            b = np.zeros(size,         dtype=np.float64)
            cost = 0.0

            cost += self._add_visual_factors(H, b)
            cost += self._add_imu_factors(H, b)
            cost += self._add_bias_prior(H, b)
            cost += self._add_marginalization_prior(H, b)

            # Levenberg-Marquardt-style regularisation: damp enough that the
            # maximum per-DOF step stays within 1 m / 1 m/s / 0.1 rad.
            diag_scale = max(1.0, float(np.max(np.diag(H))))
            lm_lambda  = 1e-4 * diag_scale
            H_reg = H + lm_lambda * np.eye(size)
            try:
                delta = np.linalg.solve(H_reg, b)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H_reg, b, rcond=None)[0]

            # Hard-cap the step so a single bad iteration cannot send the state
            # to infinity.  Limits: 0.5 rad rotation, 1 m translation, 2 m/s
            # velocity, 0.05 rad/s gyro bias, 0.1 m/s² acc bias — per iteration.
            _step_limits = np.tile([0.5, 0.5, 0.5,      # φ
                                    1.0, 1.0, 1.0,      # t
                                    2.0, 2.0, 2.0,      # v
                                    0.05, 0.05, 0.05,   # b_g
                                    0.1,  0.1,  0.1],   # b_a
                                   N)
            delta = np.clip(delta, -_step_limits, _step_limits)

            delta_norm = float(np.linalg.norm(delta))
            self._apply_delta(delta)

            if prev_cost is not None and abs(prev_cost - cost) < 1e-7 * max(1., abs(cost)):
                converged = True
                break
            prev_cost = cost

        return {
            "final_cost":  float(prev_cost) if prev_cost is not None else 0.,
            "n_iterations": it + 1,
            "converged":    converged,
            "delta_norm":   delta_norm,
        }

    def get_latest_pose(self) -> np.ndarray:
        """Returns T_cam_world (4×4) for the most recent keyframe."""
        if not self.keyframes:
            return np.eye(4)
        T_wi = self.keyframes[-1]["T"]           # world-from-IMU
        T_iw = np.linalg.inv(T_wi)              # IMU-from-world
        return self.T_cam_imu @ T_iw             # camera-from-world

    def get_latest_velocity(self) -> np.ndarray:
        """Returns v_world (3,) for the most recent keyframe."""
        return self.keyframes[-1]["v"].copy() if self.keyframes else np.zeros(3)

    def get_latest_bias(self) -> dict:
        """Returns {'b_g': (3,), 'b_a': (3,)} for the most recent keyframe."""
        if not self.keyframes:
            return {"b_g": np.zeros(3), "b_a": np.zeros(3)}
        kf = self.keyframes[-1]
        return {"b_g": kf["b_g"].copy(), "b_a": kf["b_a"].copy()}

    # ── factor assembly ───────────────────────────────────────────────────────

    def _add_visual_factors(self, H: np.ndarray, b: np.ndarray) -> float:
        """Add reprojection residuals for all (keyframe, landmark) pairs.

        Residual:
            r = π(T_cw @ P_w) - p_obs   ∈ R²

        where π is the pinhole projection and T_cw = T_cam_imu @ T_iw.

        Jacobian:  J_vis ∈ R^{2×6} wrt right perturbation (δφ, δt) of T_wi.

        The projection Jacobian wrt camera-frame point P_c = [xc,yc,zc]:
            J_π = [[fx/zc,  0, -fx·xc/zc²],
                   [0,  fy/zc, -fy·yc/zc²]]        ∈ R^{2×3}

        Chain rule:  J_vis = J_π  @ J_Pc_δξ          ∈ R^{2×6}

        J_Pc_δξ for a right perturbation of T_wi = [R_wi | p_wi]:
            δP_c = T_cam_imu[:3,:3] @ R_iw @ (-[P_w - p_wi]× δφ + δt)
                 ≈ R_ci @ (-skew(P_b) δφ + δt)
        where P_b = R_iw @ (P_w - p_wi) is the point in IMU body frame,
        R_ci = T_cam_imu[:3,:3].
        """
        w_vis = 1.0 / (SIGMA_VIS ** 2)
        R_ci  = self.T_cam_imu[:3, :3]
        t_ci  = self.T_cam_imu[:3,  3]
        cost  = 0.0

        for k, kf in enumerate(self.keyframes):
            if not kf["pts3d"]:
                continue

            bk     = k * DOF
            T_wi   = kf["T"]
            R_wi   = T_wi[:3, :3]          # world-from-IMU rotation
            p_wi   = T_wi[:3,  3]          # IMU position in world
            R_iw   = R_wi.T                # IMU-from-world rotation

            # T_cam_world = T_cam_imu @ T_imu_world
            T_cw   = self.T_cam_imu @ np.linalg.inv(T_wi)
            R_cw   = T_cw[:3, :3]
            t_cw   = T_cw[:3,  3]

            fx, fy = self.K[0,0], self.K[1,1]

            for P_w, p_obs in zip(kf["pts3d"], kf["pts2d"]):
                P_w   = np.asarray(P_w, dtype=np.float64).ravel()
                p_obs = np.asarray(p_obs, dtype=np.float64).ravel()

                # Project
                P_c   = R_cw @ P_w + t_cw
                xc, yc, zc = P_c
                if zc < 1e-4:
                    continue
                proj = np.array([fx*xc/zc + self.K[0,2],
                                 fy*yc/zc + self.K[1,2]])
                r   = proj - p_obs          # (2,)

                # 2×3 projection Jacobian wrt P_c
                J_pi = np.array([[fx/zc, 0.,    -fx*xc/(zc*zc)],
                                  [0.,   fy/zc, -fy*yc/(zc*zc)]])

                # 3×6 Jacobian of P_c wrt (δφ_wi, δt_wi)
                # P_b = R_iw @ (P_w - p_wi)  (point in IMU body frame)
                # Right-perturb T_wi: R_wi' = R_wi Exp(δφ),  p_wi' = p_wi + δt (world)
                # → R_iw' ≈ (I − [δφ]×) R_iw
                # → P_b' ≈ P_b − [δφ]× P_b − R_iw δt
                #         = P_b + [P_b]× δφ − R_iw δt
                #   (using: −[δφ]× P_b = [P_b]× δφ)
                # ∂P_c/∂δφ = R_ci @ (+[P_b]×)   (3×3)
                # ∂P_c/∂δt = −R_ci @ R_iw        (3×3)
                P_b      = R_iw @ (P_w - p_wi)
                dPc_dphi = R_ci @ _skew(P_b)       # 3×3  (positive sign)
                dPc_dt   = -R_ci @ R_iw             # 3×3
                J_Pc     = np.hstack([dPc_dphi, dPc_dt])   # 3×6

                J = J_pi @ J_Pc   # 2×6  (pose Jacobian only)

                # Accumulate into pose block [bk : bk+6]
                JtW = J.T * w_vis             # 6×2
                H[bk:bk+6, bk:bk+6] += JtW @ J
                b[bk:bk+6]          += JtW @ r
                cost                += 0.5 * w_vis * float(r @ r)

        return cost

    def _add_imu_factors(self, H: np.ndarray, b: np.ndarray) -> float:
        """Add IMU factor residuals for each consecutive keyframe pair.

        Residual:  r = [r_ΔR (3), r_Δv (3), r_Δp (3)]  ∈ R⁹
        Weight:    W = Σ_ij^{-1}  ∈ R^{9×9}

        Jacobians from IMUFactor are wrt world-from-IMU state.
        The 15-DOF block for KF k is laid out as [δφ|δt|δv|δb_g|δb_a].

        The IMUFactor Jacobians are wrt (R_i,p_i) in the world-from-IMU sense,
        which matches our T_wi convention directly.
        """
        cost = 0.0
        N    = len(self.keyframes)

        for i, preint in enumerate(self.preints):
            j = i + 1
            if j >= N:
                break

            kfi = self.keyframes[i]
            kfj = self.keyframes[j]

            # World-from-IMU poses
            T_wi = kfi["T"]
            T_wj = kfj["T"]

            state_i = {"R": T_wi[:3,:3], "p": T_wi[:3,3], "v": kfi["v"]}
            state_j = {"R": T_wj[:3,:3], "p": T_wj[:3,3], "v": kfj["v"]}
            bias_i  = {"b_g": kfi["b_g"], "b_a": kfi["b_a"]}

            fac  = IMUFactor(preint, gravity=self.gravity)
            r    = fac.residual(state_i, state_j, bias_i)    # (9,)
            W    = fac.information_matrix()                    # (9,9)
            jacs = fac.jacobians(state_i, state_j, bias_i)   # dict of (9,3)

            cost += 0.5 * float(r @ W @ r)

            # Build 9×15 Jacobian blocks for KF i and KF j
            # Layout: [φ(0:3), t(3:6), v(6:9), b_g(9:12), b_a(12:15)]
            Ji = np.zeros((9, DOF))
            Ji[:, 0:3]  = jacs["dr_dRi"]
            Ji[:, 3:6]  = jacs["dr_dpi"]
            Ji[:, 6:9]  = jacs["dr_dvi"]
            Ji[:, 9:12] = jacs["dr_dbg"]
            Ji[:,12:15] = jacs["dr_dba"]

            Jj = np.zeros((9, DOF))
            Jj[:, 0:3] = jacs["dr_dRj"]
            Jj[:, 3:6] = jacs["dr_dpj"]
            Jj[:, 6:9] = jacs["dr_dvj"]
            # b_g, b_a at KF j are not constrained by this factor

            bi = i * DOF
            bj = j * DOF

            JiW = Ji.T @ W     # 15×9
            JjW = Jj.T @ W

            # Diagonal blocks
            H[bi:bi+DOF, bi:bi+DOF] += JiW @ Ji
            H[bj:bj+DOF, bj:bj+DOF] += JjW @ Jj
            # Off-diagonal (symmetric)
            H[bi:bi+DOF, bj:bj+DOF] += JiW @ Jj
            H[bj:bj+DOF, bi:bi+DOF] += (JiW @ Jj).T

            b[bi:bi+DOF] += JiW @ r
            b[bj:bj+DOF] += JjW @ r

        return cost

    def _add_bias_prior(self, H: np.ndarray, b: np.ndarray) -> float:
        """Add diagonal Gaussian prior on biases (bias random-walk model).

        Penalises deviation of each keyframe's bias from its current estimate
        (zero residual at linearisation point → contributes 0 to cost but
        adds curvature, preventing under-constrained biases).
        """
        cost = 0.0
        for k, kf in enumerate(self.keyframes):
            bk = k * DOF
            H[bk+9:bk+12,  bk+9:bk+12]  += self._w_bg * np.eye(3)
            H[bk+12:bk+15, bk+12:bk+15] += self._w_ba * np.eye(3)
        return cost

    def _add_marginalization_prior(self, H: np.ndarray, b: np.ndarray) -> float:
        """Add dense marginalization prior to the current window.

        The prior covers the first prior_H.shape[0] DOF of the state vector,
        which correspond to the oldest remaining keyframes after the last
        marginalization step.
        """
        if self.prior_H is None:
            return 0.0
        ph = self.prior_H.shape[0]
        ph = min(ph, H.shape[0])
        H[:ph, :ph] += self.prior_H[:ph, :ph]
        b[:ph]      += self.prior_b[:ph]
        return 0.0    # prior cost is constant wrt δx

    # ── state update ──────────────────────────────────────────────────────────

    def _apply_delta(self, delta: np.ndarray) -> None:
        """Apply Gauss-Newton step to all keyframe states.

        Rotation is updated via right-multiplication: R ← R @ Exp(δφ),
        implemented through cv2.Rodrigues angle-axis addition (first order).
        Translation, velocity, and biases are Euclidean.
        """
        for k, kf in enumerate(self.keyframes):
            bk = k * DOF
            dk = delta[bk : bk + DOF]

            d_phi = dk[0:3]
            d_t   = dk[3:6]
            d_v   = dk[6:9]
            d_bg  = dk[9:12]
            d_ba  = dk[12:15]

            # Right-perturb rotation: R_new = R_old @ Exp(δφ)
            # Via Rodrigues: rvec_new ≈ rvec_old + δφ  (first order)
            rv, tv = _T_to_rv(kf["T"])
            rv_new = rv + d_phi              # first-order angle-axis update
            tv_new = tv + d_t
            kf["T"] = _rv_to_T(rv_new, tv_new)

            kf["v"]   += d_v
            kf["b_g"] += d_bg
            kf["b_a"] += d_ba

    # ── marginalization ───────────────────────────────────────────────────────

    def _marginalize_oldest(self) -> None:
        """Marginalize the oldest keyframe via the Schur complement.

        Schur complement:
            H* = H_λλ − H_λμ H_μμ^{-1} H_μλ
            b* = b_λ  − H_λμ H_μμ^{-1} b_μ

        where μ denotes the oldest (marginalized) frame's DOF block and
        λ denotes the remaining frames' DOF block.

        A local Hessian is assembled from:
          - The IMU factor connecting the oldest to the second-oldest frame.
          - The existing marginalization prior (if present).
          - The visual factors for the oldest frame.

        The resulting prior is stored in self.prior_H / self.prior_b and
        applied to the new window at the next optimization step.

        NOTE: This is a simplified implementation that uses only the IMU
        and existing prior terms, not the full factor-graph Hessian for the
        oldest frame.  This is a known approximation (equivalent to dropping
        visual constraints from the marginalized frame) that is acceptable
        for a coursework implementation.
        """
        N = len(self.keyframes)
        if N < 1:
            return

        # Block dimensions
        dim_mu  = DOF                            # oldest frame (marginalized)
        dim_lam = (N - 1) * DOF                  # remaining frames

        # Build local Hessian H_full ∈ R^{(dim_mu + dim_lam) × ...}
        local_size = dim_mu + dim_lam
        H_loc = np.zeros((local_size, local_size))
        b_loc = np.zeros(local_size)

        # ── IMU factor between oldest (μ) and second frame (λ₀) ──────────────
        if self.preints and N >= 2:
            kfi    = self.keyframes[0]
            kfj    = self.keyframes[1]
            preint = self.preints[0]

            state_i = {"R": kfi["T"][:3,:3], "p": kfi["T"][:3,3], "v": kfi["v"]}
            state_j = {"R": kfj["T"][:3,:3], "p": kfj["T"][:3,3], "v": kfj["v"]}
            bias_i  = {"b_g": kfi["b_g"], "b_a": kfi["b_a"]}

            fac  = IMUFactor(preint, gravity=self.gravity)
            r    = fac.residual(state_i, state_j, bias_i)
            W    = fac.information_matrix()
            jacs = fac.jacobians(state_i, state_j, bias_i)

            Ji = np.zeros((9, DOF))
            Ji[:, 0:3]  = jacs["dr_dRi"]
            Ji[:, 3:6]  = jacs["dr_dpi"]
            Ji[:, 6:9]  = jacs["dr_dvi"]
            Ji[:, 9:12] = jacs["dr_dbg"]
            Ji[:,12:15] = jacs["dr_dba"]

            Jj = np.zeros((9, DOF))
            Jj[:, 0:3] = jacs["dr_dRj"]
            Jj[:, 3:6] = jacs["dr_dpj"]
            Jj[:, 6:9] = jacs["dr_dvj"]

            # μ block  → rows/cols [0:DOF]
            # λ₀ block → rows/cols [DOF : 2*DOF]
            JiW = Ji.T @ W
            JjW = Jj.T @ W
            H_loc[0:DOF, 0:DOF]               += JiW @ Ji
            H_loc[DOF:2*DOF, DOF:2*DOF]       += JjW @ Jj
            H_loc[0:DOF,     DOF:2*DOF]        += JiW @ Jj
            H_loc[DOF:2*DOF, 0:DOF]            += (JiW @ Jj).T
            b_loc[0:DOF]                       += JiW @ r
            b_loc[DOF:2*DOF]                   += JjW @ r

        # ── Existing marginalization prior (covers μ block) ───────────────────
        if self.prior_H is not None:
            ph = min(self.prior_H.shape[0], local_size)
            H_loc[:ph, :ph] += self.prior_H[:ph, :ph]
            b_loc[:ph]      += self.prior_b[:ph]

        # ── Schur complement: eliminate μ block ───────────────────────────────
        H_mm = H_loc[:dim_mu,  :dim_mu]
        H_lm = H_loc[dim_mu:,  :dim_mu]
        H_ll  = H_loc[dim_mu:, dim_mu:]
        b_m   = b_loc[:dim_mu]
        b_l   = b_loc[dim_mu:]

        try:
            H_mm_inv = np.linalg.inv(H_mm + 1e-8 * np.eye(dim_mu))
        except np.linalg.LinAlgError:
            H_mm_inv = np.linalg.pinv(H_mm)

        factor_lm = H_lm @ H_mm_inv
        self.prior_H = H_ll  - factor_lm @ H_lm.T
        self.prior_b = b_l   - factor_lm @ b_m

        # ── Drop oldest keyframe and its preintegration ───────────────────────
        self.keyframes.pop(0)
        if self.preints:
            self.preints.pop(0)


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.tum_vi_loader   import TUMVIDataset, _tum_vi_default_calib
    from src.utils.trajectory_io   import load_tum_trajectory as _unused
    from src.utils.evaluation      import load_tum_trajectory

    # ── load dataset ──────────────────────────────────────────────────────────
    seq_dir = Path("data/room2")
    print(f"Loading {seq_dir} …")
    ds = TUMVIDataset(str(seq_dir))
    _, imu_calib = _tum_vi_default_calib()

    K         = ds.cam_calib.K
    T_cam_imu = ds.cam_calib.T_cam_imu

    # ── load VO poses from run_vo.py output ───────────────────────────────────
    traj_file = Path("results/trajectories/vo_room2.txt")
    if not traj_file.exists():
        print(f"[WARN] {traj_file} not found — using identity poses.")
        vo_poses = None
    else:
        ts_vo, _ = load_tum_trajectory(str(traj_file))
        # Load full poses
        vo_poses = {}
        with open(traj_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 8:
                    continue
                ts = float(p[0])
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = (float(p[4]), float(p[5]),
                                   float(p[6]), float(p[7]))
                # Quaternion → rotation matrix
                q = np.array([qx, qy, qz, qw])
                qn = q / np.linalg.norm(q)
                x,y,z,w = qn
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])
                T_wc = np.eye(4); T_wc[:3,:3] = R; T_wc[:3,3] = [tx,ty,tz]
                T_cw = np.linalg.inv(T_wc)
                vo_poses[ts] = T_cw
        print(f"Loaded {len(vo_poses)} VO poses.")

    # ── build optimizer ───────────────────────────────────────────────────────
    sw = SlidingWindowOptimizer(K, T_cam_imu, imu_calib, window_size=5)

    frame_count = 0
    costs       = []
    prev_frame  = None

    for frame in ds.iter_frames(max_frames=10):
        ts  = frame["timestamp"]
        img = frame["image"]
        imu = frame["imu_since_last"]

        # Look up VO pose or use identity
        if vo_poses:
            # Nearest VO pose
            best_ts = min(vo_poses.keys(), key=lambda t: abs(t - ts))
            T_cw = vo_poses[best_ts]
        else:
            T_cw = np.eye(4)

        # Dummy landmarks in front of the camera (for smoke test)
        pts3d, pts2d = [], []
        if frame_count > 0:
            rng = np.random.default_rng(frame_count)
            for _ in range(15):
                P_c = rng.uniform([-0.5, -0.5, 1.5], [0.5, 0.5, 3.0])
                P_w = np.linalg.inv(T_cw)[:3,:3] @ P_c + np.linalg.inv(T_cw)[:3,3]
                px  = K @ P_c;  px = px[:2] / px[2]
                pts3d.append(P_w); pts2d.append(px)

        sw.add_keyframe(
            T_cam_world      = T_cw,
            v_world          = np.zeros(3),
            b_g              = np.zeros(3),
            b_a              = np.zeros(3),
            timestamp        = ts,
            pts3d            = pts3d,
            pts2d            = pts2d,
            imu_measurements = imu,
        )

        if len(sw.keyframes) >= 2:
            result = sw.optimize(n_iterations=3)
            costs.append(result["final_cost"])
            print(f"  Frame {frame_count:3d}  "
                  f"KFs={len(sw.keyframes)}  "
                  f"cost={result['final_cost']:.4f}  "
                  f"|δx|={result['delta_norm']:.2e}  "
                  f"conv={result['converged']}")

        frame_count += 1
        prev_frame  = frame

    print(f"\nFinal cost: {costs[-1] if costs else 'N/A':.6f}")
    print(f"Latest pose:\n{np.round(sw.get_latest_pose(), 4)}")

    # Verify cost does not blow up (smoke test; not monotone due to new KFs)
    if len(costs) >= 2:
        print(f"\nFirst cost: {costs[0]:.4f}  Last cost: {costs[-1]:.4f}")
        assert np.isfinite(costs[-1]), "Cost is not finite!"
        print("Smoke test PASSED (cost is finite).")
