"""Entry point: run VIO (VO + IMU) on a TUM VI sequence.

Usage: python run_vio.py --seq data/room2 --config configs/room2.yaml
"""
import argparse, numpy as np
np.random.seed(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",    required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(f"[run_vio] seq={args.seq}  config={args.config}")
    # TODO: wire up pipeline

if __name__ == "__main__":
    main()
