"""ORB / SIFT feature detection and ratio-test matching."""

import cv2
import numpy as np
from pathlib import Path


class FeatureDetector:
    """Wraps ORB or SIFT detection and BFMatcher for the VO front-end."""

    def __init__(self, method: str = "ORB", n_features: int = 1000, seed: int = 0):
        np.random.seed(seed)
        self.method = method.upper()
        if self.method == "ORB":
            self._detector = cv2.ORB_create(nfeatures=n_features)
            self._norm = cv2.NORM_HAMMING
        elif self.method == "SIFT":
            self._detector = cv2.SIFT_create(nfeatures=n_features)
            self._norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'ORB' or 'SIFT'.")
        self._matcher = cv2.BFMatcher(self._norm)

    def detect(self, img: np.ndarray):
        """Detect keypoints and compute descriptors on a grayscale image.

        First stage of the VO front-end: extracts local features from each
        incoming frame so that cross-frame correspondences can be established
        for ego-motion estimation.

        Returns (keypoints, descriptors).
        """
        keypoints, descriptors = self._detector.detectAndCompute(img, None)
        return keypoints, descriptors

    def match(self, desc1, desc2, kp1, kp2,
              ratio_thresh: float = 0.75, cross_check: bool = True):
        """Match descriptors between two frames using Lowe's ratio test.

        Core data-association step: establishes 2-D pixel correspondences
        that feed essential-matrix estimation or PnP solving. The ratio test
        discards ambiguous matches; the optional mutual cross-check further
        prunes outliers without a geometric model.

        Returns (pts1, pts2) as float32 arrays of shape (N, 2).
        """
        def _ratio_filtered(matches_2nn):
            good = {}
            for pair in matches_2nn:
                if len(pair) < 2:
                    continue
                m, n = pair[0], pair[1]
                if m.distance < ratio_thresh * n.distance:
                    good[m.queryIdx] = m.trainIdx
            return good

        fwd = _ratio_filtered(self._matcher.knnMatch(desc1, desc2, k=2))

        if cross_check:
            rev = _ratio_filtered(self._matcher.knnMatch(desc2, desc1, k=2))
            pairs = [(i, j) for i, j in fwd.items() if rev.get(j) == i]
        else:
            pairs = list(fwd.items())

        if not pairs:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        pts1 = np.array([kp1[i].pt for i, _ in pairs], dtype=np.float32)
        pts2 = np.array([kp2[j].pt for _, j in pairs], dtype=np.float32)
        return pts1, pts2

    def draw_matches(self, img1, img2, kp1, kp2, pts1, pts2) -> np.ndarray:
        """Render matched point pairs as a side-by-side debug image.

        Visualises which pixels in frame 1 correspond to which pixels in
        frame 2 so that match quality can be inspected before the geometric
        verification stage (essential matrix / RANSAC).

        Returns a BGR image with match lines drawn.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

        left  = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
        right = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
        canvas[:h1, :w1] = left
        canvas[:h2, w1:] = right

        rng = np.random.default_rng(seed=42)
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            color = tuple(int(c) for c in rng.integers(100, 255, 3))
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w1, int(y2)),
                     color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
            cv2.circle(canvas, (int(x2) + w1, int(y2)), 3, color, -1)

        return canvas


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from src.utils.tum_vi_loader import TUMVIDataset

    ds = TUMVIDataset(str(ROOT / "data" / "room2"))

    frames = []
    for frame in ds.iter_frames(max_frames=2):
        frames.append(frame)

    img0, img1 = frames[0]["image"], frames[1]["image"]

    det = FeatureDetector(method="ORB", n_features=1000)
    kp0, desc0 = det.detect(img0)
    kp1, desc1 = det.detect(img1)
    pts0, pts1 = det.match(desc0, desc1, kp0, kp1)

    print(f"Keypoints frame 0 : {len(kp0)}")
    print(f"Keypoints frame 1 : {len(kp1)}")
    print(f"Matches after ratio test + cross-check: {len(pts0)}")

    out_dir = ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = det.draw_matches(img0, img1, kp0, kp1, pts0, pts1)
    out_path = out_dir / "match_debug.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved → {out_path}")
