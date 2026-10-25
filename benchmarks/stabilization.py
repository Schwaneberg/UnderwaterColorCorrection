"""Measure background motion on comparison exports using optional OpenCV.

Use the same grade for both files. This is a diagnostic, not a ground-truth
stabilization score: moving subjects, parallax and poor texture can bias motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from uwcc.media import FrameReader, probe, proxy_size


def measure(path, roi=(.6, .65, .95, .95)):
    import cv2
    cv2.setNumThreads(1)
    motions, inlier_counts = [], []
    previous = None
    info = probe(path)
    width, _ = proxy_size(info, 640)
    with FrameReader(info, max_width=640, sample_rate=25) as reader:
        for frame in reader:
            gray = cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            if previous is not None:
                mask = np.zeros_like(gray)
                x0, y0, x1, y1 = (int(roi[0]*width), int(roi[1]*height),
                                   int(roi[2]*width), int(roi[3]*height))
                mask[y0:y1, x0:x1] = 255
                points = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=.005,
                                                minDistance=5, mask=mask, blockSize=7)
                motion = [float("nan")] * 3
                count = 0
                if points is not None and len(points) >= 6:
                    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None,
                                                               winSize=(21, 21), maxLevel=3)
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, tracked, None,
                                                                 winSize=(21, 21), maxLevel=3)
                    valid = (status.ravel() > 0) & (back_status.ravel() > 0)
                    valid &= np.linalg.norm(points - back, axis=(1, 2)) < 1
                    if valid.sum() >= 6:
                        matrix, inliers = cv2.estimateAffinePartial2D(
                            points[valid], tracked[valid], method=cv2.RANSAC,
                            ransacReprojThreshold=1, maxIters=1000)
                        if matrix is not None and inliers is not None:
                            count = int(inliers.sum())
                            if count >= 6 and count >= valid.sum()*.5:
                                center = np.array([(x0+x1)/2, (y0+y1)/2, 1])
                                shift = matrix @ center - center[:2]
                                motion = [float(shift[0]), float(shift[1]),
                                          float(np.arctan2(matrix[1, 0], matrix[0, 0]))]
                motions.append(motion)
                inlier_counts.append(count)
            previous = gray
    values = np.asarray(motions, dtype=float).reshape(-1, 3)
    valid = np.isfinite(values).all(axis=1)
    # Differences of inter-frame displacement suppress a steady intentional pan.
    # Only compare adjacent valid estimates; never bridge tracking failures.
    adjacent = valid[1:] & valid[:-1]
    acceleration = np.diff(values[:, :2], axis=0)[adjacent]
    return {"path": str(Path(path).resolve()), "roi_normalized": list(roi), "proxy_width": width,
            "sample_rate": 25, "frame_pairs": len(values), "valid_pairs": int(valid.sum()),
            "median_inliers": float(np.median(np.asarray(inlier_counts)[valid])) if valid.any() else None,
            "translation_acceleration_rms_proxy_pixels":
                float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1)))) if len(acceleration) else None,
            "motion": [[float(v) if np.isfinite(v) else None for v in row] for row in values]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original")
    parser.add_argument("stabilized")
    parser.add_argument("--roi", nargs=4, type=float, default=(.6, .65, .95, .95),
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error("ROI must be an ordered rectangle with coordinates between 0 and 1")
    result = {"original": measure(args.original, args.roi), "stabilized": measure(args.stabilized, args.roi)}
    original = np.asarray(result["original"]["motion"], dtype=float).reshape(-1, 3)
    stabilized = np.asarray(result["stabilized"]["motion"], dtype=float).reshape(-1, 3)
    if original.shape != stabilized.shape:
        parser.error("Comparison clips must produce the same number of sampled frames")
    valid = np.isfinite(original).all(axis=1) & np.isfinite(stabilized).all(axis=1)
    adjacent = valid[1:] & valid[:-1]
    paired = {"adjacent_pairs": int(adjacent.sum())}
    for label, values in (("original", original), ("stabilized", stabilized)):
        delta = np.diff(values[:, :2], axis=0)[adjacent]
        paired[label + "_acceleration_rms_proxy_pixels"] = (
            float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))) if len(delta) else None)
    result["paired_comparison"] = paired
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    for label in ("original", "stabilized"):
        row = result[label]
        print(f"{label}: {row['valid_pairs']}/{row['frame_pairs']} tracked pairs; "
              f"background acceleration RMS {row['translation_acceleration_rms_proxy_pixels']} proxy pixels")
    print("Same-frame comparison:", json.dumps(paired))


if __name__ == "__main__":
    main()
