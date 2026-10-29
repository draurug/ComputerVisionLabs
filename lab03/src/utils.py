# common stuff: own Harris, non-maximum suppression, point matching

import csv
import glob
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N_POINTS = 1000   # same limit for every detector
NMS_RADIUS = 10   # same suppression radius for every detector
EPS = 3           # px, same threshold for the repeatability


def image_paths():
    return sorted(glob.glob(os.path.join(DATA, "*.jpg")))


def harris_response(image, k=0.04, sigma=1.5, box=None):
    # steps 1-5; box = window size to sum over a square window instead of the gaussian
    # (that is what cv2.cornerHarris does with blockSize)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # products first, then smoothing
    if box is None:
        smooth = lambda m: cv2.GaussianBlur(m, (0, 0), sigma)
    else:
        smooth = lambda m: cv2.boxFilter(m, -1, (box, box), normalize=False)
    a = smooth(ix * ix)
    b = smooth(iy * iy)
    c = smooth(ix * iy)
    return (a * b - c * c) - k * (a + b) ** 2


def nms(points, scores, radius=NMS_RADIUS, n=N_POINTS):
    # steps 6-7: go through the points in order of decreasing score, keep a point if it is
    # farther than radius from everything kept so far, stop at n points
    order = np.argsort(-scores)
    points = points[order]
    kept = []
    for p in points:
        if kept:
            d = np.linalg.norm(np.array(kept) - p, axis=1)
            if d.min() <= radius:
                continue
        kept.append(p)
        if len(kept) == n:
            break
    return np.array(kept, dtype=np.float32)


def points_from_response(r, fraction=0.01):
    # threshold by a fraction of the maximum, and keep only local maxima (3x3)
    # so that the greedy loop in nms has fewer candidates
    local_max = cv2.dilate(r, np.ones((3, 3), np.uint8))
    ys, xs = np.where((r > fraction * r.max()) & (r == local_max))
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    return nms(points, r[ys, xs])


def harris_points(image, box=None):
    return points_from_response(harris_response(image, box=box))


def keypoints_to_points(keypoints):
    # same limit and the same nms as for harris, by the detector's own response
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    resp = np.array([kp.response for kp in keypoints], dtype=np.float32)
    if len(pts) == 0:
        return pts
    return nms(pts, resp)


def matched_fraction(p1, p2, eps=EPS):
    # share of p1 points that have a p2 point closer than eps
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    dist, _ = cKDTree(p2).query(p1)
    return np.mean(dist < eps)


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
