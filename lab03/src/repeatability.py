# Lab 3, part D: repeatability of Harris, SIFT and ORB under known transformations

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import EPS, FIG, N_POINTS, harris_points, image_paths, keypoints_to_points, save_csv

np.random.seed(42)

sift = cv2.SIFT_create(nfeatures=N_POINTS)
orb = cv2.ORB_create(nfeatures=N_POINTS)


def detect(name, image):
    if name == "Harris":
        return harris_points(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = sift if name == "SIFT" else orb
    return keypoints_to_points(det.detect(gray, None))


def transform(image, kind, value):
    # returns the new image and the 3x3 matrix H that maps original points to it
    h, w = image.shape[:2]
    if kind == "rotation":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), value, 1.0)
        H = np.vstack([m, [0, 0, 1]])
        return cv2.warpPerspective(image, H, (w, h)), H
    if kind == "scale":
        H = np.diag([value, value, 1.0])
        return cv2.warpPerspective(image, H, (int(w * value), int(h * value))), H
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8), np.eye(3)
    if kind == "noise":
        noisy = image.astype(np.float32) + np.random.normal(0, value, image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8), np.eye(3)


def repeatability(p1, p2, H, shape):
    # Step 15: move p1 with H, drop the ones that fell outside the new image,
    # count how many got a p2 point closer than EPS, divide by min(|P1|, |P2|)
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    moved = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = shape[:2]
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < w) & (moved[:, 1] >= 0) & (moved[:, 1] < h)
    moved = moved[inside]
    if len(moved) == 0:
        return 0.0
    d = np.linalg.norm(moved[:, None, :] - p2[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    # each p2 point can be used only once, otherwise two close p1 points count the same p2 twice
    matched = np.unique(nearest[d.min(axis=1) < EPS])
    return len(matched) / min(len(moved), len(p2))


# Step 16. Grid of experiments
grid = {
    "rotation": np.arange(0, 91, 10),
    "scale": np.arange(0.5, 2.01, 0.25),
    "brightness": np.arange(0.5, 1.51, 0.1).round(2),
    "noise": np.arange(0, 21, 5),
}
detectors = ["Harris", "SIFT", "ORB"]

images = [cv2.imread(p) for p in image_paths()]
originals = {d: [detect(d, im) for im in images] for d in detectors}

rows = []
for kind, values in grid.items():
    print(kind)
    mean = {d: [] for d in detectors}
    for v in values:
        for d in detectors:
            scores = []
            for im, p1 in zip(images, originals[d]):
                warped, H = transform(im, kind, v)
                p2 = detect(d, warped)
                scores.append(repeatability(p1, p2, H, warped.shape))
            mean[d].append(np.mean(scores))
            rows.append([kind, v, d, round(np.mean(scores), 4), round(np.std(scores), 4)])
        print(f"  {v}: " + "  ".join(f"{d} {mean[d][-1] * 100:.0f} %" for d in detectors))

    plt.figure(figsize=(6, 4))
    for d in detectors:
        plt.plot(values, np.array(mean[d]) * 100, "o-", label=d)
    plt.xlabel({"rotation": "angle, deg", "scale": "scale", "brightness": "brightness factor",
                "noise": "noise sigma"}[kind])
    plt.ylabel("repeatability, %")
    plt.ylim(0, 105)
    plt.title(f"{kind}, mean over {len(images)} images")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(FIG, f"repeatability_{kind}.png"), dpi=150, bbox_inches="tight")
    plt.close()

save_csv("repeatability.csv", ["transform", "value", "detector", "mean", "std"], rows)
