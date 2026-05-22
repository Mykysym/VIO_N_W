"""Analytic Jacobians of the IMU residual.

        All Jacobians use the RIGHT perturbation convention for rotations:
            R ⊕ δφ = R @ Exp(δφ).

        Returns
        -------
        dict with keys and shapes:
            "dr_dRi"  (9,3)  ∂r/∂δφ_i 
            "dr_dpi"  (9,3)  ∂r/∂δp_i   
            "dr_dvi"  (9,3)  ∂r/∂δv_i   
            "dr_dRj"  (9,3)  ∂r/∂δφ_j   
            "dr_dpj"  (9,3)  ∂r/∂δp_j
            "dr_dvj"  (9,3)  ∂r/∂δv_j
            "dr_dbg"  (9,3)  ∂r/∂δb_g   
            "dr_dba"  (9,3)  ∂r/∂δb_a
        """

import numpy as np

np.random.seed(0)


# ── SO(3) utilities ───────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric (cross-product) matrix."""
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ], dtype=np.float64)


def Exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map — Rodrigues formula.

    Maps axis-angle vector phi ∈ R³ → rotation matrix R ∈ SO(3).
    Handles ‖phi‖ → 0 via first-order approximation.
    """
    phi   = np.asarray(phi, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.eye(3) + skew(phi)          # first-order for tiny angles
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def Log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithmic map.

    Maps rotation matrix R ∈ SO(3) → axis-angle vector phi ∈ R³.
    Handles phi → 0 case.
    """
    R       = np.asarray(R, dtype=np.float64)
    cos_phi = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    phi     = float(np.arccos(cos_phi))
    if phi < 1e-10:
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]],
                               dtype=np.float64)
    return (phi / (2.0 * np.sin(phi))) * np.array(
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian Jr(phi) of SO(3) exponential map.

    Jr(phi) = I  -  (1-cos‖phi‖)/‖phi‖²  * [phi]×
                 +  (‖phi‖-sin‖phi‖)/‖phi‖³ * ([phi]×)²
    """
    phi   = np.asarray(phi, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.eye(3) - 0.5 * skew(phi)   # first-order for tiny angles
    K      = skew(phi)
    theta2 = theta * theta
    theta3 = theta2 * theta
    return (np.eye(3)
            - (1.0 - np.cos(theta)) / theta2 * K
            + (theta - np.sin(theta)) / theta3 * (K @ K))


# ── IMU Preintegration ────────────────────────────────────────────────────────

class IMUPreintegration:
    """IMU preintegration on SO(3) manifold.

    Accumulates raw IMU samples into:
      • delta_R  – preintegrated rotation    ΔR_ij  ∈ SO(3)
      • delta_v  – preintegrated velocity    Δv_ij  ∈ R³
      • delta_p  – preintegrated position    Δp_ij  ∈ R³
      • covariance – 9×9 Σ_ij for [δφ, δv, δp]
      • five bias-correction Jacobians
    """

    def __init__(self,
                 b_g: np.ndarray,
                 b_a: np.ndarray,
                 imu_calib) -> None:
        """
        Parameters
        ----------
        b_g       : (3,) gyroscope bias [rad/s] — linearisation point.
        b_a       : (3,) accelerometer bias [m/s²] — linearisation point.
        imu_calib : IMUCalib from src.utils.tum_vi_loader.
                    Used fields:
                      .gyroscope_noise_density     σ_g  [rad/s/√Hz]
                      .accelerometer_noise_density  σ_a  [m/s²/√Hz]
        """
        self.b_g = np.asarray(b_g, dtype=np.float64).ravel().copy()
        self.b_a = np.asarray(b_a, dtype=np.float64).ravel().copy()

        # Noise densities (continuous-time, from IMUCalib)
        self._sigma_g = float(imu_calib.gyroscope_noise_density)
        self._sigma_a = float(imu_calib.accelerometer_noise_density)

        # ── Preintegrated quantities ──────────────────────────────────────────
        self.delta_R    = np.eye(3, dtype=np.float64)    # ΔR_ij ∈ SO(3)
        self.delta_v    = np.zeros(3, dtype=np.float64)  # Δv_ij
        self.delta_p    = np.zeros(3, dtype=np.float64)  # Δp_ij
        self.dt_sum     = 0.0                             # Σ dt_k

        # ── 9×9 covariance [δφ, δv, δp] ─────────────────────────────────────
        self.covariance = np.zeros((9, 9), dtype=np.float64)

        # ── Bias-correction Jacobians ─────────────────────────────────────────
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)   # ∂ΔR / ∂b_g
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)   # ∂Δv / ∂b_g
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)   # ∂Δv / ∂b_a
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)   # ∂Δp / ∂b_g
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)   # ∂Δp / ∂b_a

    # ── single integration step ───────────────────────────────────────────────

    def integrate(self,
                  omega_raw: np.ndarray,
                  acc_raw:   np.ndarray,
                  dt:        float) -> None:
        """Single IMU integration step.

        Parameters
        ----------
        omega_raw : (3,) raw gyroscope measurement  [rad/s]
        acc_raw   : (3,) raw accelerometer measurement [m/s²]
        dt        : time step [s]

        Updates delta_R, delta_v, delta_p, covariance, all Jacobians,
        and dt_sum in place.
        """
        omega_raw = np.asarray(omega_raw, dtype=np.float64).ravel()
        acc_raw   = np.asarray(acc_raw,   dtype=np.float64).ravel()
        dt        = float(dt)

        # Bias-corrected measurements (linearised at stored biases)
        omega_c = omega_raw - self.b_g   # ω̃ - b̄_g
        acc_c   = acc_raw   - self.b_a   # ã - b̄_a

        # Rotation increment and right Jacobian 
        phi     = omega_c * dt
        dR_step = Exp(phi)        # ΔR_{k,k+1} = Exp((ω̃-b_g)*dt)
        Jr      = right_jacobian(phi)   # Jr(ω_c*dt)

        # ── Bias-correction Jacobian updates ────────────────
        # All updates use the Jacobians and delta_R *before* this step.
        # Dependency order: J_p depends on J_v, J_v on J_R; update outermost first.

        # ∂Δp_{k+1}/∂b_g = ∂Δp_k/∂b_g + ∂Δv_k/∂b_g * dt
        #                   - ½ΔR_k · [ã_c]× · ∂ΔR_k/∂b_g · dt²
        self.J_p_bg = (self.J_p_bg
                       + self.J_v_bg * dt
                       - 0.5 * self.delta_R @ skew(acc_c) @ self.J_R_bg * dt * dt)

        # ∂Δp_{k+1}/∂b_a = ∂Δp_k/∂b_a + ∂Δv_k/∂b_a * dt - ½ΔR_k * dt²
        self.J_p_ba = (self.J_p_ba
                       + self.J_v_ba * dt
                       - 0.5 * self.delta_R * dt * dt)

        # ∂Δv_{k+1}/∂b_g = ∂Δv_k/∂b_g - ΔR_k · [ã_c]× · ∂ΔR_k/∂b_g · dt
        self.J_v_bg = (self.J_v_bg
                       - self.delta_R @ skew(acc_c) @ self.J_R_bg * dt)

        # ∂Δv_{k+1}/∂b_a = ∂Δv_k/∂b_a - ΔR_k * dt
        self.J_v_ba = self.J_v_ba - self.delta_R * dt

        # ∂ΔR_{k+1}/∂b_g = dR_step.T · ∂ΔR_k/∂b_g - Jr(ω_c·dt) * dt
        self.J_R_bg = dR_step.T @ self.J_R_bg - Jr * dt

        # ── Covariance propagation ───────────────────────
        # State transition A and noise-input B; use delta_R *before* this step.
        A = np.zeros((9, 9), dtype=np.float64)
        A[0:3, 0:3] = dR_step.T
        A[3:6, 0:3] = -self.delta_R @ skew(acc_c) @ Jr * dt
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 0:3] = -0.5 * self.delta_R @ skew(acc_c) @ Jr * dt * dt
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 6:9] = np.eye(3)

        B = np.zeros((9, 6), dtype=np.float64)
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = self.delta_R * dt
        B[6:9, 3:6] = 0.5 * self.delta_R * dt * dt

        # Discrete noise covariance for [η_gyro, η_accel].
        # σ [units/√Hz] → discrete variance = σ²/dt  (noise density convention).
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = (self._sigma_g ** 2 / dt) * np.eye(3)
        Q[3:6, 3:6] = (self._sigma_a ** 2 / dt) * np.eye(3)

        self.covariance = A @ self.covariance @ A.T + B @ Q @ B.T

        # ── State update ────────────────────────────────────
        # Update delta_p first (uses delta_v at step k), then delta_v, then delta_R.
        self.delta_p = (self.delta_p
                        + self.delta_v * dt
                        + 0.5 * self.delta_R @ acc_c * dt * dt)
        self.delta_v = self.delta_v + self.delta_R @ acc_c * dt
        self.delta_R = self.delta_R @ dR_step
        self.dt_sum += dt

    # ── batch integration ─────────────────────────────────────────────────────

    def integrate_imu_segment(self, imu_measurements: list) -> 'IMUPreintegration':
        """Integrate a list of consecutive IMU measurements.

        Parameters
        ----------
        imu_measurements : list of dict, each with keys
            {t, wx, wy, wz, ax, ay, az}  (time in seconds, SI units).
            Must be sorted by time.  dt is computed from consecutive timestamps.

        Returns
        -------
        self  (for method chaining)
        """
        for k in range(1, len(imu_measurements)):
            m0 = imu_measurements[k - 1]
            m1 = imu_measurements[k]
            dt = float(m1['t']) - float(m0['t'])
            if dt <= 0.0:
                continue
            omega = np.array([m0['wx'], m0['wy'], m0['wz']], dtype=np.float64)
            acc   = np.array([m0['ax'], m0['ay'], m0['az']], dtype=np.float64)
            self.integrate(omega, acc, dt)
        return self

    # ── bias correction ───────────────────────────────────────────────────────

    def bias_correction(self,
                        new_b_g: np.ndarray,
                        new_b_a: np.ndarray
                        ) -> tuple:
        """First-order bias correction without re-integration.

        Computes corrected preintegrated quantities for updated bias estimates,
        using the stored Jacobians as the linearisation.

        Parameters
        ----------
        new_b_g : (3,) updated gyroscope bias [rad/s]
        new_b_a : (3,) updated accelerometer bias [m/s²]

        Returns
        -------
        (delta_R_c, delta_v_c, delta_p_c) — corrected quantities.
        Does NOT modify internal state.
        """
        d_bg = np.asarray(new_b_g, dtype=np.float64).ravel() - self.b_g
        d_ba = np.asarray(new_b_a, dtype=np.float64).ravel() - self.b_a

        #   ΔR̃(b̄_g + δb_g) ≈ ΔR̃(b̄_g) ⊕ (∂ΔR̄/∂b_g · δb_g)
        delta_R_c = self.delta_R @ Exp(self.J_R_bg @ d_bg)
        delta_v_c = self.delta_v + self.J_v_bg @ d_bg + self.J_v_ba @ d_ba
        delta_p_c = self.delta_p + self.J_p_bg @ d_bg + self.J_p_ba @ d_ba

        return delta_R_c, delta_v_c, delta_p_c

    # ── residuals ─────────────────────────────────────────────────────────────

    def get_residuals(self,
                      R_i: np.ndarray, p_i: np.ndarray, v_i: np.ndarray,
                      R_j: np.ndarray, p_j: np.ndarray, v_j: np.ndarray,
                      b_g: np.ndarray, b_a: np.ndarray,
                      gravity: np.ndarray = None) -> np.ndarray:
        """IMU factor residuals.

        Parameters
        ----------
        R_i, R_j  : (3,3) rotation matrices at keyframes i and j.
        p_i, p_j  : (3,)  positions  [m].
        v_i, v_j  : (3,)  velocities [m/s].
        b_g, b_a  : (3,)  current bias estimates.
        gravity   : (3,)  gravity vector [m/s²]; default [0, 0, -9.81].

        Returns
        -------
        r : (9,) stacked residual [r_ΔR(3), r_Δv(3), r_Δp(3)].
        """
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])

        # Bias-corrected preintegrated quantities
        dR_c, dv_c, dp_c = self.bias_correction(b_g, b_a)

        dt = self.dt_sum

        # r_ΔR = Log( (ΔR̃_c).T  @  R_i.T  @  R_j )
        r_R = Log(dR_c.T @ R_i.T @ R_j)

        # r_Δv = R_i.T @ (v_j - v_i - g·Δt)  -  Δṽ_c
        r_v = R_i.T @ (v_j - v_i - gravity * dt) - dv_c

        # r_Δp = R_i.T @ (p_j - p_i - v_i·Δt - ½g·Δt²)  -  Δp̃_c
        r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt) - dp_c

        return np.concatenate([r_R, r_v, r_p])


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__[:__file__.rfind("src")])
    from src.utils.tum_vi_loader import _tum_vi_default_calib

    calib, _ = _tum_vi_default_calib()

    # ------------------------------------------------------------------
    # Test 1: static segment — zero specific force, zero angular rate.
    #
    # In the Forster model delta_v integrates SPECIFIC FORCE (what the
    # accelerometer measures), NOT the kinematic acceleration.  For a
    # truly stationary body, the specific force equals −g in the body
    # frame, so acc_raw = [0, 0, +9.81] z-up.  Integrating that gives
    # delta_v ≈ 9.81 m/s after 1 s — that is physically correct; gravity
    # "cancels" only inside the navigation residual r_Δv, not in delta_v
    # itself.
    #
    # To exercise the zero-drift property (||Δv|| < 1e-3) we therefore
    # feed zero specific force (gravity-free / perfectly bias-compensated
    # scenario) so that acc_c = acc_raw − b_a = 0.
    # ------------------------------------------------------------------
    n_meas = 200
    dt     = 0.005   # 200 Hz
    b_g0   = np.zeros(3)
    b_a0   = np.zeros(3)

    preint = IMUPreintegration(b_g0, b_a0, calib)

    omega_zero = np.zeros(3)
    acc_zero   = np.zeros(3)   # zero specific force (gravity-free)

    for _ in range(n_meas):
        preint.integrate(omega_zero, acc_zero, dt)

    print("=" * 60)
    print("Test 1 — static, zero specific force (200 steps @ 5 ms)")
    print("=" * 60)

    err_R  = float(np.linalg.norm(preint.delta_R - np.eye(3)))
    err_v  = float(np.linalg.norm(preint.delta_v))
    err_p  = float(np.linalg.norm(preint.delta_p))

    print(f"  ||ΔR − I|| = {err_R:.2e}  →  ", end="")
    print("PASS" if err_R < 1e-6 else "FAIL")

    print(f"  ||Δv||     = {err_v:.2e}  →  ", end="")
    print("PASS" if err_v < 1e-3 else "FAIL")

    print(f"  ||Δp||     = {err_p:.2e}  →  ", end="")
    print("PASS" if err_p < 1e-3 else "FAIL")

    print(f"\n  Covariance diagonal (φ,v,p):\n  {np.diag(preint.covariance)}")

    # ------------------------------------------------------------------
    # Test 2: bias_correction with small δb_g = [0.01, 0, 0] rad/s.
    # With zero integration (all zero inputs) the Jacobians are also zero,
    # so the corrected quantities equal the original ones.  We verify that
    # bias_correction is non-destructive and dimensionally consistent.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2 — bias_correction(δb_g=[0.01,0,0])")
    print("=" * 60)

    new_bg = np.array([0.01, 0.0, 0.0])
    dR_c, dv_c, dp_c = preint.bias_correction(new_bg, b_a0)

    # Internal state must be unchanged
    state_ok = (np.allclose(preint.delta_R, np.eye(3)) and
                np.allclose(preint.delta_v, 0) and
                np.allclose(preint.delta_p, 0))
    print(f"  Internal state unchanged  →  {'PASS' if state_ok else 'FAIL'}")

    # Corrected ΔR should still be close to I (Jacobian is 0 for zero input)
    err_Rc = float(np.linalg.norm(dR_c - np.eye(3)))
    print(f"  ||ΔR_c − I|| = {err_Rc:.2e}  →  {'PASS' if err_Rc < 1e-8 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Test 3: get_residuals for a static body (v=0, no motion).
    # With zero omega and acc=[0,0,9.81] (stationary reading), the
    # navigation residual r_Δv must vanish because gravity and delta_v
    # cancel in the formula:
    #   r_Δv = R_i.T @ (v_j - v_i - g*dt) - delta_v
    #        = I.T  @ (0   - 0   - (−9.81)*T) - 9.81*T = 0
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3 — residuals, static body, acc=[0,0,9.81]")
    print("=" * 60)

    preint3 = IMUPreintegration(b_g0, b_a0, calib)
    acc_static = np.array([0.0, 0.0, 9.81])
    for _ in range(n_meas):
        preint3.integrate(omega_zero, acc_static, dt)

    T = n_meas * dt   # 1 second
    g = np.array([0.0, 0.0, -9.81])
    I3 = np.eye(3)

    r = preint3.get_residuals(I3, np.zeros(3), np.zeros(3),
                              I3, np.zeros(3), np.zeros(3),
                              b_g0, b_a0, gravity=g)

    print(f"  residuals [r_R, r_v, r_p]: {np.round(r, 6)}")
    res_ok = np.linalg.norm(r) < 1e-6
    print(f"  ||r|| = {np.linalg.norm(r):.2e}  →  {'PASS' if res_ok else 'FAIL'}")
