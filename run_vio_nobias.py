"""VIO ablation: frozen (zero-initialised) IMU bias.

Thin wrapper around run_vio.  The SlidingWindowOptimizer is replaced with a
subclass that zeroes the b_g / b_a update blocks each Gauss-Newton step, so
both biases stay at zero throughout optimisation.  Every other part of the
pipeline — feature tracking, initialisation, keyframe selection, IMU
propagation — is bit-for-bit identical to run_vio.py.

Output trajectory:  results/trajectories/vio_{seq}_nobias.txt
Output plot:        results/plots/trajectories/vio_{seq}_nobias.png

Usage:
    python run_vio_nobias.py --seq data/room2 --config configs/room2.yaml

Purpose:
    Ablation study isolating the effect of online IMU bias estimation.
"""

import numpy as np

np.random.seed(0)

from pathlib import Path

import run_vio                                          # full pipeline
from src.backend.sliding_window import SlidingWindowOptimizer, DOF


# ── fixed-bias optimizer ───────────────────────────────────────────────────────

class FixedBiasSlidingWindowOptimizer(SlidingWindowOptimizer):
    """SlidingWindowOptimizer whose bias states are never updated.

    On each Gauss-Newton step the b_g (indices 9:12) and b_a (indices 12:15)
    blocks of the delta vector are zeroed before the update is applied.  Pose,
    velocity, IMU factor residuals, and the bias-random-walk prior are all
    computed and assembled normally; only the bias *increment* is suppressed,
    so the optimiser never drifts the biases away from their initial value of
    zero.  This is the correct ablation for isolating the effect of online
    bias estimation without touching sliding_window.py.
    """

    def _apply_delta(self, delta: np.ndarray) -> None:
        for k in range(len(self.keyframes)):
            bk = k * DOF
            delta[bk + 9  : bk + 12] = 0.0   # freeze Δb_g
            delta[bk + 12 : bk + 15] = 0.0   # freeze Δb_a
        super()._apply_delta(delta)


# ── thin main wrapper ──────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Swap in the fixed-bias optimizer ───────────────────────────────────
    # Python resolves global names at call time, so patching the module
    # attribute before run_vio.main() is entered makes both SlidingWindowOptimizer
    # construction sites (initial build + post-init rebuild) use the subclass.
    run_vio.SlidingWindowOptimizer = FixedBiasSlidingWindowOptimizer

    # ── 2. Redirect trajectory output → _nobias.txt ───────────────────────────
    _orig_save = run_vio.save_tum_trajectory

    def _save_nobias(poses, timestamps, path: str) -> None:
        p = Path(path)
        if p.stem.startswith("vio_") and "_nobias" not in p.stem:
            path = str(p.parent / (p.stem + "_nobias" + p.suffix))
        _orig_save(poses, timestamps, path)

    run_vio.save_tum_trajectory = _save_nobias

    # ── 3. Redirect plot output → _nobias.png ─────────────────────────────────
    # plt is the shared matplotlib.pyplot module object; we patch savefig on it
    # and restore in the finally block so no other caller is affected.
    import matplotlib.pyplot as plt

    _orig_savefig = plt.savefig

    def _savefig_nobias(path, **kwargs) -> None:
        if isinstance(path, str):
            p = Path(path)
            if p.stem.startswith("vio_") and "_nobias" not in p.stem:
                path = str(p.parent / (p.stem + "_nobias" + p.suffix))
        _orig_savefig(path, **kwargs)

    plt.savefig = _savefig_nobias

    # ── 4. Run the full pipeline ───────────────────────────────────────────────
    try:
        run_vio.main()
    finally:
        plt.savefig = _orig_savefig     # always restore, even on error


if __name__ == "__main__":
    main()
