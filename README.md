# Monocular VO / VIO on TUM VI

Classical geometry-based **Visual Odometry (VO)** and **Visual-Inertial Odometry (VIO)** implemented from scratch in Python, evaluated on the [TUM VI benchmark](https://vision.in.tum.de/data/datasets/visual-inertial-dataset).

The system estimates 6-DoF camera trajectories using only a single camera (VO) or camera + IMU (VIO). No deep learning is used — the pipeline is built entirely from epipolar geometry, IMU preintegration, and nonlinear optimisation.

---

## Features

- **Monocular VO** — feature tracking → essential matrix bootstrap → PnP + motion-only bundle adjustment
- **Metric VIO** — VINS-Mono-style initialisation (gyro bias, gravity, scale) → sliding-window visual-inertial optimisation
- **IMU preintegration on manifold** — Forster et al. RSS 2015 with bias correction Jacobians
- **Analytic Jacobians** — numerically verified against finite differences
- **Evaluation** — ATE, RPE, Umeyama alignment (Sim(3) for VO, SE(3) for VIO)
- **Ablation** — VIO without IMU bias estimation (`run_vio_nobias.py`)

---

## Requirements

```bash
pip install opencv-python numpy scipy matplotlib pyyaml
```

Python 3.10 or newer. No GPU required.

---

## Dataset Setup

Download TUM VI sequences from https://vision.in.tum.de/data/datasets/visual-inertial-dataset and place them under `data/` using the standard `mav0` export:

```
data/
├── room2/
│   └── mav0/
│       ├── cam0/data/          ← grayscale images (*.png)
│       ├── imu0/data.csv       ← IMU at 200 Hz
│       └── mocap0/data.csv     ← ground-truth poses
├── corridor3/
│   └── mav0/  ...
└── outdoors5/
    └── mav0/  ...
```

The loader also accepts the DSO-style export layout (`dso/cam0/images/`, `dso/imu.txt`). If no `camchain-imucam.yaml` calibration file is present the pipeline falls back to the TUM VI factory defaults.

---

## Quick Start

### 1. Monocular VO (arbitrary scale)

```bash
python run_vo.py --seq data/room2 --config configs/room2.yaml
```

### 2. Visual-Inertial Odometry (metric scale)

```bash
python run_vio.py --seq data/room2 --config configs/room2.yaml
```

### 3. Ablation — VIO without IMU bias estimation

```bash
python run_vio_nobias.py --seq data/room2 --config configs/room2.yaml
```

### 4. Evaluate

```bash
# Single sequence, one method
python evaluate.py --seq room2 --method vo
python evaluate.py --seq room2 --method vio

# Side-by-side VO vs VIO, all sequences
python evaluate.py --seq all --method all
```

All outputs land in `results/`:

| Output                       | Path                                           |
| ---------------------------- | ---------------------------------------------- |
| Trajectory file (TUM format) | `results/trajectories/{method}_{seq}.txt`      |
| Per-frame ATE CSV            | `results/tables/{method}_{seq}_metrics.csv`    |
| ATE / trajectory plot        | `results/plots/metrics/{method}_{seq}_ate.png` |
| RPE histogram                | `results/plots/metrics/{method}_{seq}_rpe.png` |

---

## Results

Evaluated on room2 (full ground truth). Trajectories aligned with Umeyama SVD.

| Method       | ATE RMSE | Alignment | Notes                        |
| ------------ | -------- | --------- | ---------------------------- |
| Monocular VO | ~1.2 m   | Sim(3)    | Scale recovered by alignment |
| VIO          | ~42 m    | SE(3)     | Metric scale from IMU init   |

---

## Configuration

Each sequence has a YAML config in `configs/`. Key parameters:

```yaml
# configs/room2.yaml
seq_name: room2
cam: cam0
seed: 0

vo:
  detector: ORB # ORB or SIFT
  n_features: 1000
  min_matches: 30
  ransac_threshold: 1.0 # px

vio:
  window_size: 5 # keyframes in sliding window
  gravity: [0.0, 9.81, 0.0] # m/s² — +y is down for TUM VI cam0
  min_init_frames: 30 # frames before VIO init triggers
  max_init_frames: 300 # timeout frames
  kf_parallax_thresh: 30.0 # px
  kf_min_tracks: 80
  kf_max_interval: 5 # force keyframe every N frames
  imu_weight:
    0.007 # scale IMU information matrix
    # (raw IMU info ~6700× larger than visual)

eval:
  align: se3
  rpe_segment_len: 100 # metres
```

---

## Pipeline Overview

### Monocular VO

```
Frame k
 │
 ├─[Bootstrap] Essential matrix → cheirality pose recovery → triangulate landmarks
 │             Scale fixed so median landmark depth = 1 unit
 │
 ├─[Tracking]  LK optical flow (forward-backward check ≤ 1 px)
 │             Re-detect when tracked count < min_tracks
 │
 ├─[Pose]      solvePnPRansac (200 iter, 99.9%) + solvePnPRefineLM
 │
 └─[Map]       Motion-only BA (SciPy, Huber δ=1 px) → triangulate new landmarks
```

### VIO — 3-phase architecture (VINS-Mono, Qin et al. TRO 2018)

```
Phase 1 — VO bootstrap
  Advance frames until E-matrix gives rotation < 30° and ≥ 12 valid landmarks
  Run standard VO; feed poses + IMU to VIOInitializer

VIO Init (VINS-Mono Section V, closed-form)
  Step 1: Gyroscope bias  — least-squares on rotation residuals
  Step 2: Velocity + gravity + scale — linear system from IMU kinematics
  Step 3: Validate (scale > 0, gravity magnitude within 10% of prior)
  → Apply metric scale to all landmarks and poses

Phase 2 — Metric tracking
  Per keyframe:
    IMU propagation (Forster 2015 eq. 24) → warm-start PnP
    PnP → SlidingWindowOptimizer (visual + IMU + bias prior + marginalisation)
    Accept SW output if |v| < 20 m/s and position drift from PnP < 2 m
    Otherwise: reset SW, use finite-difference velocity
  Per non-keyframe:
    IMU propagation only
```

---

## Project Structure

```
.
├── run_vo.py                   Entry point: monocular VO
├── run_vio.py                  Entry point: VIO (camera + IMU)
├── run_vio_nobias.py           Ablation: VIO with frozen IMU biases
├── evaluate.py                 ATE / RPE evaluation, plots, tables
│
├── configs/
│   ├── room2.yaml              Indoor room (full GT)
│   ├── corridor3.yaml          Long corridor (partial GT)
│   └── outdoors5.yaml          Outdoor (partial GT)
│
├── src/
│   ├── frontend/
│   │   ├── feature_detector.py ORB / SIFT detection + matching
│   │   ├── feature_tracker.py  LK optical flow, persistent track IDs
│   │   └── epipolar.py         Essential matrix, pose recovery, triangulation
│   │
│   ├── backend/
│   │   ├── pnp_solver.py           PnP RANSAC + LM refinement
│   │   ├── bundle_adjustment.py    Motion-only BA (SciPy + Huber)
│   │   ├── imu_preintegration.py   SO(3) preintegration (Forster RSS 2015)
│   │   ├── imu_factor.py           IMU residual + analytic Jacobians
│   │   ├── vio_initializer.py      VINS-Mono Section V initialisation
│   │   └── sliding_window.py       Gauss-Newton sliding-window optimiser
│   │
│   └── utils/
│       ├── tum_vi_loader.py    Dataset loader + Kalibr calibration parser
│       ├── trajectory_io.py    TUM trajectory format read/write
│       └── evaluation.py       Umeyama alignment, ATE, RPE
│
├── data/                       TUM VI sequences (not tracked)
├── results/                    Output files (not tracked)
├── README.md                   This file
└── explanation.md              Full technical documentation
```

---

## References

1. D. Schubert, T. Goll, N. Demmel, V. Usenko, J. St¨uckler, and D. Cremers,
“The TUM VI Benchmark for Evaluating Visual-Inertial Odometry,”
International Conference on Intelligent Robots and Systems (IROS), 2018.
2. A. I. Mourikis and S. I. Roumeliotis, “A Multi-State Constraint Kalman
Filter for Vision-aided Inertial Navigation,” in Proceedings 2007 IEEE
International Conference on Robotics and Automation. IEEE, 2007, pp.
3565–3572.
3. S. Leutenegger, S. Lynen, M. Bosse, R. Siegwart, and P. Furgale,
“Keyframe-based visual–inertial odometry using nonlinear optimization,”
The International Journal of Robotics Research, vol. 34, no. 3, pp. 314–
334, 2015.
4. T. Qin, P. Li, and S. Shen, “VINS-Mono: A Robust and Versatile Monocular
Visual-Inertial State Estimator,” IEEE Transactions on Robotics,
vol. 34, no. 4, pp. 1004–1020, 2018.
5. C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, “IMU Preintegration
on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori
Estimation,” in Robotics: Science and Systems (RSS), 2015.
6. E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient
alternative to SIFT or SURF,” in 2011 International conference on
computer vision. Ieee, 2011, pp. 2564–2571.
7. B. D. Lucas and T. Kanade, “An iterative image registration technique
with an application to stereo vision,” in Proceedings of imaging understanding
workshop, 1981, pp. 121–130.

---

## Reproducibility

All scripts call `np.random.seed(0)` at startup. The seed can be overridden via `seed:` in the YAML config. OpenCV RANSAC uses its own internal random state and may give slightly different results across platforms.

Hardware used: Intel Core i7-9750H CPU, Windows 11 Home 25H2. Typical runtime: ~40 ms/frame for VIO (room2).
