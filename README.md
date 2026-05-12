# VIO on TUM VI

Monocular Visual Odometry → Visual-Inertial Odometry (Master's project).

## Setup
```bash
pip install opencv-python numpy scipy matplotlib pyyaml
```

## Quickstart
```bash
# Point DATASET_ROOT at your sequence
python run_vo.py  --seq data/room2  --config configs/room2.yaml
python run_vio.py --seq data/room2  --config configs/room2.yaml
python evaluate.py --seq room2
```

## Structure
```
src/frontend/   — feature detection, tracking, epipolar geometry
src/backend/    — PnP, bundle adjustment, IMU preintegration, sliding window
src/utils/      — data loader, trajectory I/O, evaluation metrics, plots
configs/        — per-sequence YAML (paths, tuning knobs, seeds)
results/        — output trajectories, metric tables, plots (git-ignored)
tests/          — unit tests
report/         — IEEE paper source
```

## Reproducibility
All scripts call `np.random.seed(0)`. Seeds and runtimes are logged to
`results/tables/<seq>_runtime.csv`.
