"""Lucas-Kanade optical-flow tracking across consecutive frames."""

import cv2
import numpy as np
from pathlib import Path

from src.frontend.feature_detector import FeatureDetector

_DEFAULT_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class FeatureTracker:
    """Tracks 2-D feature points across frames with LK optical flow.

    Maintains a persistent set of tracks identified by integer IDs so that
    downstream modules (e.g. sliding-window BA) can associate 2-D observations
    with 3-D landmarks across an arbitrary number of frames.
    """

    def __init__(self,
                 detector: FeatureDetector,
                 min_tracks: int = 30,
                 lk_params: dict = None):
        self._detector  = detector
        self._min_tracks = min_tracks
        self._lk        = lk_params if lk_params is not None else _DEFAULT_LK

        self.prev_img:   np.ndarray | None = None
        self.prev_pts:   np.ndarray | None = None   # (N, 1, 2) float32
        self.track_ids:  np.ndarray | None = None   # (N,) int64
        self.next_id:    int = 0
        self.n_tracked:  int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def init(self, img: np.ndarray) -> None:
        """Detect initial features and seed the tracker.

        Called once before the first call to track(). All detected keypoints
        receive fresh track IDs; the frame is stored as the reference for the
        next optical-flow step.
        """
        kps, _ = self._detector.detect(img)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

        self.prev_img  = img.copy()
        self.prev_pts  = pts
        self.track_ids = self._alloc_ids(len(pts))
        self.n_tracked = len(pts)

    def track(self, img: np.ndarray):
        """Track features from the previous frame into img using LK flow.

        After the forward pass, a backward pass (img → prev) rejects any pair
        whose round-trip reprojection error exceeds 1 px (forward-backward
        consistency check). If the surviving track count drops below
        min_tracks, new keypoints are detected on the current frame and merged
        into the active set with fresh IDs.

        Returns (prev_pts, curr_pts, track_ids) — float32 (N,2) arrays and
        int64 (N,) IDs for the matched pairs, suitable for epipolar geometry.
        """
        assert self.prev_img is not None, "Call init() before track()."

        # ── forward pass ──────────────────────────────────────────────────
        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, self.prev_pts, None, **self._lk
        )
        # ── backward pass (forward-backward consistency check) ────────────
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            img, self.prev_img, curr_pts, None, **self._lk
        )

        fb_err = np.linalg.norm(
            self.prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
        )
        good = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_err <= 1.0)

        prev_good = self.prev_pts[good].reshape(-1, 1, 2)
        curr_good = curr_pts[good].reshape(-1, 1, 2)
        ids_good  = self.track_ids[good]

        # ── re-detect if too few tracks ───────────────────────────────────
        if len(curr_good) < self._min_tracks:
            curr_good, ids_good = self._add_new_features(img, curr_good, ids_good)

        # ── update state ──────────────────────────────────────────────────
        self.prev_img  = img.copy()
        self.prev_pts  = curr_good
        self.track_ids = ids_good
        self.n_tracked = len(curr_good)

        return (
            prev_good.reshape(-1, 2),
            curr_good.reshape(-1, 2),
            ids_good,
        )

    def reset(self, img: np.ndarray) -> None:
        """Full re-initialisation after tracking loss.

        Clears all existing tracks, resets the ID counter, and re-seeds the
        tracker on img as if init() were called on a fresh instance.
        """
        self.prev_img  = None
        self.prev_pts  = None
        self.track_ids = None
        self.next_id   = 0
        self.n_tracked = 0
        self.init(img)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _alloc_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.next_id += n
        return ids

    def _add_new_features(self, img, existing_pts, existing_ids):
        """Detect new keypoints on img, skip those near existing tracks."""
        kps, _ = self._detector.detect(img)
        if not kps:
            return existing_pts, existing_ids

        new_pts = np.array([kp.pt for kp in kps], dtype=np.float32)

        # Mask out positions within 5 px of any existing track
        if len(existing_pts) > 0:
            ex = existing_pts.reshape(-1, 2)
            dists = np.linalg.norm(
                new_pts[:, None, :] - ex[None, :, :], axis=2
            ).min(axis=1)
            new_pts = new_pts[dists > 5.0]

        if len(new_pts) == 0:
            return existing_pts, existing_ids

        new_ids = self._alloc_ids(len(new_pts))
        merged_pts = np.vstack([
            existing_pts.reshape(-1, 2),
            new_pts,
        ]).reshape(-1, 1, 2).astype(np.float32)
        merged_ids = np.concatenate([existing_ids, new_ids])
        return merged_pts, merged_ids


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    from src.utils.tum_vi_loader import TUMVIDataset

    ds = TUMVIDataset(str(ROOT / "data" / "room2"))

    det     = FeatureDetector(method="ORB", n_features=500)
    tracker = FeatureTracker(det, min_tracks=30)

    last_img   = None
    last_pts   = None
    last_ids   = None

    for frame in ds.iter_frames(max_frames=50):
        img = frame["image"]
        idx = frame["index"]

        if idx == 0:
            tracker.init(img)
            print(f"Frame {idx:3d} | init | tracks: {tracker.n_tracked}")
        else:
            prev_pts, curr_pts, ids = tracker.track(img)
            print(f"Frame {idx:3d} | tracks: {tracker.n_tracked:4d}")
            if idx == 49:
                last_img  = img
                last_pts  = curr_pts
                last_ids  = ids

    # ── overlay plot on frame 50 (index 49) ──────────────────────────────
    if last_img is not None:
        canvas = cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR)
        rng = np.random.default_rng(seed=0)
        unique_ids = np.unique(last_ids)
        # Assign a fixed colour per track ID
        id_to_color = {
            tid: tuple(int(c) for c in rng.integers(50, 255, 3))
            for tid in unique_ids
        }
        for pt, tid in zip(last_pts, last_ids):
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4,
                       id_to_color[tid], -1, cv2.LINE_AA)

        out_dir = ROOT / "results" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tracks_frame50.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"Saved → {out_path}")
