"""VIO Initializer.

Recovers metric scale, gravity direction, initial velocities, and gyroscope
bias from up-to-scale monocular SfM poses and IMU data.
"""

import logging
import numpy as np

from src.backend.imu_preintegration import IMUPreintegration, Exp, Log, skew

logger = logging.getLogger(__name__)


# ── Quaternion helpers [qx, qy, qz, qw] ──────────────────────────────────────

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [qx, qy, qz, qw]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse: conjugate / norm² — [qx, qy, qz, qw]."""
    q = np.asarray(q, dtype=np.float64)
    n2 = float(q @ q)
    if n2 < 1e-20:
        return np.array([0., 0., 0., 1.])
    return np.array([-q[0], -q[1], -q[2], q[3]]) / n2


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [qx, qy, qz, qw]."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                          (R[1,0]-R[0,1])*s, 0.25/s], dtype=np.float64)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s,
                          (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s], dtype=np.float64)
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s,
                          (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s], dtype=np.float64)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                          0.25*s, (R[1,0]-R[0,1])/s], dtype=np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] → 3×3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


# ── Local SO(3) helper ────────────────────────────────────────────────────────

def _inv_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian Jr^{-1}(φ) of SO(3)."""
    phi   = np.asarray(phi, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(phi))
    K     = skew(phi)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * K
    theta2 = theta * theta
    coeff  = (1.0 / theta2
               - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) + 0.5 * K + coeff * (K @ K)


# ── VIOInitializer ────────────────────────────────────────────────────────────

class VIOInitializer:
    """Closed-form VIO initialization.

    Recovers metric scale s, gravity g^{c0}, per-frame velocities, and
    gyroscope bias b_g from up-to-scale monocular SfM poses and IMU data.

    Coordinate conventions
    ----------------------
    • sfm_poses[k]  = T_{c_k ← c0}  (camera-from-world, world = c0 = first cam)
    • All navigation quantities live in the c0 (world) frame.
    • Body/IMU positions: p_b_k = R_{c0←c_k} @ t_ci + s * p_sfm_k
      where t_ci = T_cam_imu[:3,3]  and  p_sfm_k = inv(T_sfm[k])[:3,3].
    """

    def __init__(self,
                 K:                np.ndarray,
                 T_cam_imu:        np.ndarray,
                 imu_calib,
                 gravity_magnitude: float = 9.81,
                 min_frames:       int   = 10) -> None:
        self.K                 = np.asarray(K,         dtype=np.float64)
        self.T_cam_imu         = np.asarray(T_cam_imu, dtype=np.float64)
        self.imu_calib         = imu_calib
        self.gravity_magnitude = float(gravity_magnitude)
        self.min_frames        = int(min_frames)

        self.sfm_poses:      list = []   # list of 4×4 SE3
        self.sfm_timestamps: list = []   # list of float
        self.imu_segments:   list = []   # list of list[dict]
        self.initialized:    bool = False

        # Results (valid only after initialized=True)
        self._b_g          = np.zeros(3)
        self._velocities:  list = []
        self._gravity:     np.ndarray = None
        self._scale:       float = None
        self._metric_poses: list = []

    # ── public API ────────────────────────────────────────────────────────────

    def add_frame(self,
                  T_cam_world_sfm: np.ndarray,
                  timestamp:       float,
                  imu_since_last:  list) -> bool:
        """Append SfM pose, timestamp, and IMU segment; attempt init if ready.

        Returns True if initialization succeeded (self.initialized is set).
        """
        self.sfm_poses.append(np.asarray(T_cam_world_sfm, dtype=np.float64))
        self.sfm_timestamps.append(float(timestamp))
        self.imu_segments.append(list(imu_since_last))

        if len(self.sfm_poses) >= self.min_frames:
            return self._try_initialize()
        return False

    def get_initial_states(self) -> dict:
        """Return metric initial states (only valid after initialized=True)."""
        if not self.initialized:
            raise RuntimeError("VIOInitializer: not yet initialized.")
        return {
            "poses":      [T.copy() for T in self._metric_poses],
            "velocities": [v.copy() for v in self._velocities],
            "b_g":        self._b_g.copy(),
            "b_a":        np.zeros(3),
            "gravity":    self._gravity.copy(),
            "scale":      float(self._scale),
        }

    # ── internal pipeline ─────────────────────────────────────────────────────

    def _try_initialize(self) -> bool:
        N = len(self.sfm_poses)
        try:
            ok = self._run_init(N)
        except Exception as exc:
            logger.warning("VIO init exception: %s — resetting state.", exc)
            ok = False
        if not ok:
            self._reset()
        return ok

    def _reset(self) -> None:
        self.sfm_poses.clear()
        self.sfm_timestamps.clear()
        self.imu_segments.clear()
        self.initialized = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _preintegrate(self, b_g: np.ndarray) -> list:
        """Preintegrate each inter-frame IMU segment with given b_g, b_a=0.

        Returns N-1 IMUPreintegration objects: preints[k] covers (frame k, frame k+1).
        """
        b_a = np.zeros(3)
        preints = []
        for k in range(1, len(self.sfm_poses)):
            p = IMUPreintegration(b_g, b_a, self.imu_calib)
            p.integrate_imu_segment(self.imu_segments[k])
            preints.append(p)
        return preints

    def _body_rotations(self) -> list:
        """R_{c0←b_k} for every frame k.

        R_{c0←b_k} = R_{c0←c_k} @ R_{c_k←b_k}
                   = T_sfm[k][:3,:3].T  @  T_cam_imu[:3,:3]
        """
        R_c_b = self.T_cam_imu[:3, :3]
        return [T[:3, :3].T @ R_c_b for T in self.sfm_poses]

    def _sfm_cam_positions(self) -> list:
        """Up-to-scale camera positions in c0 frame: inv(T_sfm[k])[:3,3]."""
        out = []
        for T in self.sfm_poses:
            # inv([R|t]) = [R.T | -R.T @ t]
            R = T[:3, :3]
            t = T[:3,  3]
            out.append((-R.T @ t).copy())
        return out

    # ── core algorithm ────────────────────────────────────────────────────────

    def _run_init(self, N: int) -> bool:
        # ── Step 1: gyroscope bias calibration ───────────────────────────────
        #
        # Minimize:  Σ_k || Log( R_{c0,b_{k+1}}.T @ R_{c0,b_k} @ γ̂_k ) ||²
        # with first-order linearisation in δb_g:
        #   γ̂(b_g + δb_g) ≈ γ̂(b_g) Exp(J_R_bg δb_g)
        # → Jr^{-1}(r0_k) J_R_bg δb_g = −r0_k   per pair.

        b_g    = np.zeros(3)
        preints = self._preintegrate(b_g)
        R_c0_b  = self._body_rotations()

        rows_A, rows_z = [], []
        for k in range(N - 1):
            p  = preints[k]
            r0 = Log(R_c0_b[k+1].T @ R_c0_b[k] @ p.delta_R)
            rows_A.append(_inv_right_jacobian(r0) @ p.J_R_bg)
            rows_z.append(-r0)

        A_bg  = np.vstack(rows_A)           # (3*(N-1), 3)
        z_bg  = np.concatenate(rows_z)      # (3*(N-1),)
        b_g, _, _, _ = np.linalg.lstsq(A_bg, z_bg, rcond=None)
        b_g   = b_g.copy()
        logger.debug("Step 1 b_g = %s", b_g)

        # Re-preintegrate with corrected gyro bias
        preints = self._preintegrate(b_g)

        # ── Step 2: velocity, gravity, scale ─────────────────────────────────
        #
        # Unknowns:  x = [ v_{b0}(3), …, v_{b_{N-1}}(3), g^{c0}(3), s(1) ]
        #            total: N*3 + 3 + 1 = N*3 + 4
        #
        # Per pair (k, k+1) — two sets of 3 equations each:
        #
        #   Velocity:
        #     v_{k+1} − v_k + g·dt = R_{c0,b_k} Δv_k
        #
        #   Position (derived from p_{b_{k+1}} = p_{b_k} + v_k·dt − ½g·dt² + R·Δp):
        #     v_k·dt − ½g·dt² − s·(p_sfm_{k+1}−p_sfm_k)
        #       = (R_{c0,c_{k+1}}−R_{c0,c_k})·t_ci − R_{c0,b_k}·Δp_k

        n_unk = N * 3 + 4
        n_eq  = (N - 1) * 6

        H = np.zeros((n_eq, n_unk))
        z = np.zeros(n_eq)

        t_ci   = self.T_cam_imu[:3, 3]
        R_c0_c = [T[:3, :3].T for T in self.sfm_poses]   # R_{c0←c_k}
        p_sfm  = self._sfm_cam_positions()

        # Normalise inter-frame SFM displacements so the scale column of H is
        # comparable in magnitude to the velocity column (dt ≈ 0.05 s).
        # Without this, a tiny initial baseline (sfm_scale ≪ 1) makes dp_sfm
        # orders of magnitude smaller than dt, ill-conditioning the least-squares
        # system and causing wildly wrong velocity estimates that diverge the IMU
        # propagation by several metres per frame.
        _disp_norms  = [float(np.linalg.norm(p_sfm[k + 1] - p_sfm[k]))
                        for k in range(N - 1)]
        _nz          = [d for d in _disp_norms if d > 1e-8]
        _sfm_norm    = max(float(np.median(_nz)) if _nz else 1.0, 1e-8)
        p_sfm_n = [p / _sfm_norm for p in p_sfm]   # normalised; s_n = s * _sfm_norm

        col_g  = N * 3       # columns N*3 : N*3+3   → gravity (3)
        col_s  = N * 3 + 3   # column  N*3+3         → scale   (1)

        for k in range(N - 1):
            p  = preints[k]
            dt = p.dt_sum
            if dt < 1e-7:
                continue

            R_c0_bk = R_c0_b[k]
            rv  = k * 6        # row offset: velocity block
            rp  = k * 6 + 3   # row offset: position block
            cv  = k * 3        # col offset: v_bk
            cv1 = (k + 1) * 3  # col offset: v_b{k+1}

            # — velocity equation —
            # v_{k+1} − v_k − g·dt = R·Δv   →   H·x = z
            H[rv:rv+3, cv:cv+3]        = -np.eye(3)
            H[rv:rv+3, cv1:cv1+3]      =  np.eye(3)
            H[rv:rv+3, col_g:col_g+3]  = -dt * np.eye(3)   # sign was wrong (+dt)
            z[rv:rv+3] = R_c0_bk @ p.delta_v

            # — position equation (uses normalised dp_sfm) —
            dp_sfm  = p_sfm_n[k+1] - p_sfm_n[k]
            d_lever = (R_c0_c[k+1] - R_c0_c[k]) @ t_ci

            H[rp:rp+3, cv:cv+3]        =  dt * np.eye(3)
            H[rp:rp+3, col_g:col_g+3]  = +0.5 * dt * dt * np.eye(3)   # sign was −
            H[rp:rp+3, col_s]          = -dp_sfm            # (3,) into column
            z[rp:rp+3] = d_lever - R_c0_bk @ p.delta_p

        x_sol, _, rank_H, _ = np.linalg.lstsq(H, z, rcond=None)

        if rank_H < n_unk:
            logger.warning("VIO init: under-determined system (rank %d / %d).",
                           rank_H, n_unk)

        velocities = [x_sol[k*3:(k+1)*3].copy() for k in range(N)]
        g_vec      = x_sol[col_g:col_g+3].copy()
        s          = float(x_sol[col_s]) / _sfm_norm   # un-normalise scale

        # ── Step 3: validation and metric pose assembly ───────────────────────
        g_norm = float(np.linalg.norm(g_vec))

        if s <= 0.0:
            logger.warning("VIO init: recovered scale s=%.4f ≤ 0 — rejecting.", s)
            return False

        if s < 0.01 or s > 1000.0:
            logger.warning("VIO init: scale s=%.4f outside (0.01, 1000) — rejecting.", s)
            return False

        tol = 0.1 * self.gravity_magnitude
        if abs(g_norm - self.gravity_magnitude) > tol:
            logger.warning("VIO init: gravity magnitude %.4f outside "
                           "[%.4f, %.4f] — rejecting.",
                           g_norm,
                           self.gravity_magnitude - tol,
                           self.gravity_magnitude + tol)
            return False

        # Build metric world-from-IMU (T_wb) poses.
        # Scale SfM translation by s, then:  T_wb = inv(T_scaled) @ T_cam_imu
        metric_poses = []
        for T_sfm in self.sfm_poses:
            T_scaled = T_sfm.copy()
            T_scaled[:3, 3] *= s
            T_wb = np.linalg.inv(T_scaled) @ self.T_cam_imu
            metric_poses.append(T_wb)

        self._b_g          = b_g
        self._velocities   = velocities
        self._gravity      = g_vec
        self._scale        = s
        self._metric_poses = metric_poses
        self.initialized   = True
        logger.info("VIO initialized: scale=%.4f  |g|=%.4f  b_g=%s",
                    s, g_norm, b_g)
        return True


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s: %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.tum_vi_loader import TUMVIDataset, _tum_vi_default_calib

    seq_dir = "data/room2"
    print(f"Loading {seq_dir} …")
    ds = TUMVIDataset(seq_dir)
    _, imu_calib = _tum_vi_default_calib()

    K         = ds.cam_calib.K
    T_cam_imu = ds.cam_calib.T_cam_imu

    init = VIOInitializer(K, T_cam_imu, imu_calib,
                          gravity_magnitude=9.81, min_frames=10)

    # Use GT poses as a proxy for monocular SfM (GT is metric → s ≈ 1).
    # Convert GT T_{world←body} → T_{cam←world}, then express relative to c0.
    T0_cam_world = None
    frames_fed   = 0

    for frame in ds.iter_frames(max_frames=15):
        gt  = frame["gt_pose"]         # T_{world←body} or None
        imu = frame["imu_since_last"]
        ts  = frame["timestamp"]

        if gt is None:
            print(f"  [WARN] frame {frame['index']} has no GT — skipping.")
            continue

        # T_{cam←world} = T_{cam←body} @ T_{body←world} = T_cam_imu @ inv(GT)
        T_cam_world = T_cam_imu @ np.linalg.inv(gt)

        # Express relative to the first camera frame (world = c0)
        if T0_cam_world is None:
            T0_cam_world = T_cam_world.copy()

        T_rel = T_cam_world @ np.linalg.inv(T0_cam_world)

        frames_fed += 1
        done = init.add_frame(T_rel, ts, imu)
        if done:
            print(f"  Initialization triggered at frame {frame['index']} "
                  f"(fed {frames_fed} frames).")
            break

    if not init.initialized:
        print("Initialization did not converge — check dataset path and GT availability.")
        sys.exit(1)

    states = init.get_initial_states()
    scale  = states["scale"]
    g_mag  = float(np.linalg.norm(states["gravity"]))
    b_g    = states["b_g"]

    print(f"\nRecovered scale    : {scale:.6f}")
    print(f"Gravity magnitude  : {g_mag:.6f}  m/s^2")
    print(f"Initial gyro bias  : {b_g}")

    assert 0.1 < scale < 10,   f"Scale {scale:.4f} outside (0.1, 10)"
    assert 8.5 < g_mag < 10.5, f"Gravity magnitude {g_mag:.4f} outside (8.5, 10.5)"
    print("\nAll assertions passed.")
