"""IMU Factor — analytic Jacobians of the IMU residual."""

import numpy as np

from src.backend.imu_preintegration import (
    Exp, Log, skew, right_jacobian, IMUPreintegration,
)


# ── SO(3) helper: inverse right Jacobian ─────────────────────────────────────

def inverse_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3), Jr^{-1}(φ).

    Jr^{-1}(φ) = I
               + ½ [φ]×
               + (1/‖φ‖² − (1+cos‖φ‖)/(2‖φ‖ sin‖φ‖)) · ([φ]×)²

    For ‖φ‖ → 0: Jr^{-1}(φ) ≈ I + ½[φ]×  (first-order approximation).
    """
    phi   = np.asarray(phi, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(phi))
    K     = skew(phi)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * K
    theta2 = theta * theta
    coeff  = (1.0 / theta2
              - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) + 0.5 * K + coeff * (K @ K)


# ── IMU Factor ────────────────────────────────────────────────────────────────

class IMUFactor:
    """Analytic Jacobians of the IMU residual.

    The residual r(x_i, x_j, b_i) ∈ R⁹ is:

        r_ΔR = Log(ΔR̃_c.T  @ R_i.T @ R_j) 
        r_Δv = R_i.T @ (v_j − v_i − g·Δt) − Δṽ_c
        r_Δp = R_i.T @ (p_j − p_i − v_i·Δt − ½g·Δt²) − Δp̃_c

    where ΔR̃_c, Δṽ_c, Δp̃_c are the bias-corrected preintegrated
    measurements."""

    def __init__(self,
                 preint:  IMUPreintegration,
                 gravity: np.ndarray = None) -> None:
        """
        Parameters
        ----------
        preint  : completed IMUPreintegration (dt_sum > 0).
        gravity : (3,) gravitational acceleration [m/s²]; default [0,0,−9.81].
        """
        self.preint  = preint
        self.gravity = (np.array([0.0, 0.0, -9.81])
                        if gravity is None
                        else np.asarray(gravity, dtype=np.float64).ravel())

    # ── residual ──────────────────────────────────────────────────────────────

    def residual(self,
                 state_i: dict,
                 state_j: dict,
                 bias_i:  dict) -> np.ndarray:
        """Compute the 9-vector IMU residual.

        Parameters
        ----------
        state_i, state_j : dicts with keys "R" (3,3), "p" (3,), "v" (3,).
        bias_i           : dict with keys "b_g" (3,), "b_a" (3,).

        Returns
        -------
        r : (9,) stacked [r_ΔR, r_Δv, r_Δp].
        """
        return self.preint.get_residuals(
            state_i["R"], state_i["p"], state_i["v"],
            state_j["R"], state_j["p"], state_j["v"],
            bias_i["b_g"], bias_i["b_a"],
            self.gravity,
        )

    # ── analytic Jacobians ────────────────────────────────────────────────────

    def jacobians(self,
                  state_i: dict,
                  state_j: dict,
                  bias_i:  dict) -> dict:
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
        R_i = np.asarray(state_i["R"], dtype=np.float64)
        p_i = np.asarray(state_i["p"], dtype=np.float64).ravel()
        v_i = np.asarray(state_i["v"], dtype=np.float64).ravel()
        R_j = np.asarray(state_j["R"], dtype=np.float64)
        p_j = np.asarray(state_j["p"], dtype=np.float64).ravel()
        v_j = np.asarray(state_j["v"], dtype=np.float64).ravel()
        b_g = np.asarray(bias_i["b_g"], dtype=np.float64).ravel()
        b_a = np.asarray(bias_i["b_a"], dtype=np.float64).ravel()

        dt = self.preint.dt_sum
        g  = self.gravity

        # Bias-corrected preintegrated quantities 
        dR_c, dv_c, dp_c = self.preint.bias_correction(b_g, b_a)

        # Rotation residual and its inverse right Jacobian
        r_R     = Log(dR_c.T @ R_i.T @ R_j)
        Jr_inv  = inverse_right_jacobian(r_R)     # Jr^{-1}(r_ΔR)

        # Frequently used quantities
        RiT  = R_i.T
        f_v  = v_j - v_i - g * dt
        f_p  = p_j - p_i - v_i * dt - 0.5 * g * dt * dt

        # ── ∂r / ∂δφ_i  ────────────────────────────────────
        #
        # r_ΔR perturbed: Log(ΔR̃_c.T @ Exp(−δφ_i) @ R_i.T @ R_j)
        #               ≈ r_ΔR − Jr^{-1}(r_ΔR) @ ΔR̃_c.T @ δφ_i
        # → ∂r_ΔR/∂δφ_i = −Jr^{-1}(r_ΔR) @ ΔR̃_c.T
        #
        # r_Δv/Δp perturbed: (I − [δφ]×) @ R_i.T @ f_v ≈ … + [R_i.T f_v]× δφ_i
        # → ∂r_Δv/∂δφ_i = [R_i.T (v_j − v_i − g Δt)]× 
        # → ∂r_Δp/∂δφ_i = [R_i.T (p_j − p_i − v_i Δt − ½g Δt²)]×
        dr_dRi = np.zeros((9, 3))
        dr_dRi[0:3, :] = -Jr_inv @ dR_c.T
        dr_dRi[3:6, :] =  skew(RiT @ f_v)
        dr_dRi[6:9, :] =  skew(RiT @ f_p)

        # ── ∂r / ∂δp_i ───────────────────────────────
        # r_ΔR, r_Δv: no p_i dependence.
        # r_Δp = R_i.T (p_j − p_i − …)  →  ∂r_Δp/∂δp_i = −R_i.T
        dr_dpi = np.zeros((9, 3))
        dr_dpi[6:9, :] = -RiT

        # ── ∂r / ∂δv_i ───────────────────────────────
        # r_Δv = R_i.T (v_j − v_i − …)   →  ∂r_Δv/∂δv_i = −R_i.T
        # r_Δp = R_i.T (… − v_i Δt − …)  →  ∂r_Δp/∂δv_i = −R_i.T Δt
        dr_dvi = np.zeros((9, 3))
        dr_dvi[3:6, :] = -RiT
        dr_dvi[6:9, :] = -RiT * dt

        # ── ∂r / ∂δφ_j ────────────────────────────────────
        # r_ΔR perturbed: Log(ΔR̃_c.T R_i.T R_j Exp(δφ_j)) ≈ r_ΔR + Jr^{-1} δφ_j
        # → ∂r_ΔR/∂δφ_j = Jr^{-1}(r_ΔR)
        dr_dRj = np.zeros((9, 3))
        dr_dRj[0:3, :] = Jr_inv

        # ── ∂r / ∂δp_j ───────────────────────────────────────────────────────
        # r_Δp = R_i.T (p_j − …)  →  ∂r_Δp/∂δp_j = R_i.T
        dr_dpj = np.zeros((9, 3))
        dr_dpj[6:9, :] = RiT

        # ── ∂r / ∂δv_j ───────────────────────────────────────────────────────
        # r_Δv = R_i.T (v_j − …)  →  ∂r_Δv/∂δv_j = R_i.T
        dr_dvj = np.zeros((9, 3))
        dr_dvj[3:6, :] = RiT

        # ── ∂r / ∂δb_g ────────────────────────────────────
        # From bias correction (eq. 36):
        #   ΔR̃_c(b_g+δb_g) ≈ ΔR̃(b_g) @ Exp(J_R_bg @ δb_g)
        #   → ΔR̃_c.T ≈ Exp(−J_R_bg δb_g) @ ΔR̃.T
        # Inserting into r_ΔR gives  ∂r_ΔR/∂δb_g = −Jr^{-1}(r_ΔR) @ J_R_bg
        # Δṽ_c += J_v_bg δb_g  → ∂r_Δv/∂δb_g = −J_v_bg
        # Δp̃_c += J_p_bg δb_g  → ∂r_Δp/∂δb_g = −J_p_bg
        dr_dbg = np.zeros((9, 3))
        dr_dbg[0:3, :] = -Jr_inv @ self.preint.J_R_bg
        dr_dbg[3:6, :] = -self.preint.J_v_bg
        dr_dbg[6:9, :] = -self.preint.J_p_bg

        # ── ∂r / ∂δb_a ───────────────────────────────────────────────────────
        # Rotation is independent of b_a to first order (J_R_ba = 0).
        # Δṽ_c += J_v_ba δb_a  → ∂r_Δv/∂δb_a = −J_v_ba
        # Δp̃_c += J_p_ba δb_a  → ∂r_Δp/∂δb_a = −J_p_ba
        dr_dba = np.zeros((9, 3))
        dr_dba[3:6, :] = -self.preint.J_v_ba
        dr_dba[6:9, :] = -self.preint.J_p_ba

        return {
            "dr_dRi": dr_dRi,
            "dr_dpi": dr_dpi,
            "dr_dvi": dr_dvi,
            "dr_dRj": dr_dRj,
            "dr_dpj": dr_dpj,
            "dr_dvj": dr_dvj,
            "dr_dbg": dr_dbg,
            "dr_dba": dr_dba,
        }

    # ── information matrix ────────────────────────────────────────────────────

    def information_matrix(self) -> np.ndarray:
        """Weight matrix Σ_ij^{-1} from the preintegration covariance.

        Returns
        -------
        W : (9,9) positive-definite information matrix.
            Falls back to pseudo-inverse if Σ is (near-)singular.
        """
        Sigma = self.preint.covariance
        # Add a tiny ridge for numerical safety when dt_sum is very small
        Sigma_reg = Sigma + 1e-12 * np.eye(9)
        try:
            return np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Sigma_reg)

    # ── numerical Jacobian check ──────────────────────────────────────────────

    def check_jacobians_numerically(self,
                                    state_i: dict,
                                    state_j: dict,
                                    bias_i:  dict,
                                    eps:     float = 1e-6) -> dict:
        """Finite-difference verification of all Jacobian blocks.

        For each 3-DoF perturbation variable, perturbs each component by ±eps
        and computes (r(+) − r₀) / eps.  Rotation states are perturbed via
        right multiplication:  R → R @ Exp(eps * eₖ).

        Parameters
        ----------
        state_i, state_j : keyframe states.
        bias_i           : bias state.
        eps              : finite-difference step.

        Returns
        -------
        errors : dict mapping Jacobian key → max absolute error.
        """
        r0       = self.residual(state_i, state_j, bias_i)
        analytic = self.jacobians(state_i, state_j, bias_i)

        def _fd_rot(which: str) -> np.ndarray:
            J = np.zeros((9, 3))
            R = state_i["R"] if which == "i" else state_j["R"]
            for c in range(3):
                dv = np.zeros(3); dv[c] = eps
                R_p = R @ Exp(dv)
                if which == "i":
                    r_p = self.residual({**state_i, "R": R_p}, state_j, bias_i)
                else:
                    r_p = self.residual(state_i, {**state_j, "R": R_p}, bias_i)
                J[:, c] = (r_p - r0) / eps
            return J

        def _fd_vec(which: str, key: str) -> np.ndarray:
            J   = np.zeros((9, 3))
            ref = (state_i if which == "i" else state_j)[key]
            for c in range(3):
                dv = np.zeros(3); dv[c] = eps
                if which == "i":
                    r_p = self.residual({**state_i, key: ref + dv}, state_j, bias_i)
                else:
                    r_p = self.residual(state_i, {**state_j, key: ref + dv}, bias_i)
                J[:, c] = (r_p - r0) / eps
            return J

        def _fd_bias(key: str) -> np.ndarray:
            J   = np.zeros((9, 3))
            ref = bias_i[key]
            for c in range(3):
                dv  = np.zeros(3); dv[c] = eps
                r_p = self.residual(state_i, state_j, {**bias_i, key: ref + dv})
                J[:, c] = (r_p - r0) / eps
            return J

        fd = {
            "dr_dRi": _fd_rot("i"),
            "dr_dpi": _fd_vec("i", "p"),
            "dr_dvi": _fd_vec("i", "v"),
            "dr_dRj": _fd_rot("j"),
            "dr_dpj": _fd_vec("j", "p"),
            "dr_dvj": _fd_vec("j", "v"),
            "dr_dbg": _fd_bias("b_g"),
            "dr_dba": _fd_bias("b_a"),
        }

        # Print comparison table
        hdr = f"{'Block':<12} {'|Analytic|':>12} {'|FD|':>12} {'max|Δ|':>12}"
        print("\n" + hdr)
        print("-" * len(hdr))
        errors = {}
        for key in analytic:
            err = float(np.max(np.abs(analytic[key] - fd[key])))
            errors[key] = err
            print(f"{key:<12} {np.linalg.norm(analytic[key]):>12.6f} "
                  f"{np.linalg.norm(fd[key]):>12.6f} {err:>12.2e}")
        return errors


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.tum_vi_loader import TUMVIDataset, _tum_vi_default_calib

    print("Loading room2 dataset …")
    ds = TUMVIDataset("data/room2")
    _, imu_calib = _tum_vi_default_calib()

    # Collect first two frames
    frames = []
    for frame in ds.iter_frames(max_frames=2):
        frames.append(frame)
    f0, f1 = frames[0], frames[1]

    # Build and integrate preintegration
    b_g0 = np.zeros(3)
    b_a0 = np.zeros(3)
    preint = IMUPreintegration(b_g0, b_a0, imu_calib)
    preint.integrate_imu_segment(f1["imu_since_last"])

    print(f"dt_sum = {preint.dt_sum:.6f} s")
    print(f"ΔR =\n{np.round(preint.delta_R, 6)}")
    print(f"|Δv| = {np.linalg.norm(preint.delta_v):.6f}  "
          f"|Δp| = {np.linalg.norm(preint.delta_p):.6f}")

    # States: identity at i; use ΔR/Δv/Δp as a rough j prediction
    state_i = {"R": np.eye(3),
                "p": np.zeros(3),
                "v": np.array([0.1, 0.05, -0.05])}   # non-zero for richer test
    state_j = {"R": preint.delta_R.copy(),
                "p": preint.delta_p.copy(),
                "v": preint.delta_v.copy()}
    bias_i  = {"b_g": b_g0.copy(), "b_a": b_a0.copy()}

    factor = IMUFactor(preint)

    print("\n=== Jacobian check (analytic vs finite difference) ===")
    errors = factor.check_jacobians_numerically(state_i, state_j, bias_i, eps=1e-6)

    print("\n--- Results ---")
    all_pass = True
    for key, err in errors.items():
        ok = err < 1e-4
        all_pass = all_pass and ok
        print(f"  {key:<12}  max_error = {err:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    assert all_pass, "One or more Jacobian blocks exceed the 1e-4 tolerance."
