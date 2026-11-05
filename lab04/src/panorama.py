# Lab 4, parts D and E: panorama from a series of frames, and the cases where it breaks

import os

import cv2
import numpy as np

from utils import FIG, features, load, match_points, ratio_matches, reprojection_error, save_csv

np.random.seed(42)
TAU = 3


def pair_homography(img_a, img_b):
    # H maps points of img_a into img_b
    kp_a, d_a = features(img_a)
    kp_b, d_b = features(img_b)
    matches = ratio_matches(d_a, d_b, 0.8)
    src, dst = match_points(kp_a, kp_b, matches)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, TAU)
    inl = mask.ravel().astype(bool)
    err = reprojection_error(H, src, dst)[inl].mean()
    return H, len(matches), inl.sum(), err


def stitch(names, tag):
    images = [load(n) for n in names]
    h, w = images[0].shape[:2]
    n = len(images)
    center = n // 2

    # Step 13. Homographies of neighbours, then chain them to the central frame
    pair = {}
    rows = []
    for i in range(n - 1):
        H, cand, inl, err = pair_homography(images[i], images[i + 1])
        pair[i] = H   # frame i -> frame i+1
        rows.append([names[i], names[i + 1], cand, inl, round(err, 3)])
        print(f"  {names[i]} -> {names[i + 1]}: {cand} candidates, {inl} inliers, error {err:.2f} px")

    to_center = [np.eye(3) for _ in range(n)]
    for i in range(center - 1, -1, -1):          # frames left of the center: i -> i+1 -> ... -> center
        to_center[i] = to_center[i + 1] @ pair[i]
    for i in range(center + 1, n):                # frames right of the center: i -> i-1 -> ... -> center
        to_center[i] = to_center[i - 1] @ np.linalg.inv(pair[i - 1])

    # Step 14. Canvas size from the moved corners of every frame
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = np.vstack([cv2.perspectiveTransform(corners, H) for H in to_center]).reshape(-1, 2)
    x_min, y_min = np.floor(moved.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(moved.max(axis=0)).astype(int)
    offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    size = (x_max - x_min, y_max - y_min)
    print(f"  canvas {size[0]} x {size[1]}")

    # Step 15. Warp every frame and blend with weights that go to zero at the frame border,
    # this gives a linear transition in the overlap
    canvas = np.zeros((size[1], size[0], 3), dtype=np.float32)
    weight_sum = np.zeros((size[1], size[0]), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    frame_weight = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32) + 1
    for img, H in zip(images, to_center):
        warped = cv2.warpPerspective(img.astype(np.float32), offset @ H, size)
        wgt = cv2.warpPerspective(frame_weight, offset @ H, size)
        canvas += warped * wgt[:, :, None]
        weight_sum += wgt
    canvas = canvas / np.maximum(weight_sum, 1e-6)[:, :, None]
    cv2.imwrite(os.path.join(FIG, f"panorama_{tag}.jpg"), canvas.astype(np.uint8))

    # Step 16. Quality: mean error on the inliers over all pairs
    mean_err = np.mean([r[4] for r in rows])
    print(f"  mean reprojection error over the pairs: {mean_err:.3f} px")
    save_csv(f"panorama_{tag}.csv", ["frame a", "frame b", "candidates", "inliers", "mean error on inliers, px"],
             rows + [["all pairs", "", "", "", round(mean_err, 3)]])
    return mean_err


# Part D. Pure rotation of the camera, 5 frames
print("panorama from the rotation frames")
stitch(["rot_01.jpg", "rot_02.jpg", "rot_03.jpg", "rot_04.jpg", "rot_05.jpg"], "rotation")

# Part E. Failure cases
# 1. parallax: at the end of the video a pole is close to the camera, the frames are not related by
#    a single homography any more
print("\nparallax: frames with the pole near the camera")
stitch(["rot_07.jpg", "rot_08.jpg", "rot_09.jpg", "rot_10.jpg"], "parallax")

# 2. small overlap: frames far apart in the series
print("\nsmall overlap")
rows = []
for a, b in [("rot_03.jpg", "rot_04.jpg"), ("rot_03.jpg", "rot_05.jpg"), ("rot_03.jpg", "rot_06.jpg"),
             ("rot_03.jpg", "rot_07.jpg"), ("rot_01.jpg", "rot_07.jpg")]:
    img_a, img_b = load(a), load(b)
    try:
        H, cand, inl, err = pair_homography(img_a, img_b)
    except Exception:
        H, cand, inl, err = None, 0, 0, float("nan")
    print(f"  {a} - {b}: {cand} candidates, {inl} inliers, error {err:.2f} px")
    rows.append([a, b, cand, inl, round(err, 3) if err == err else ""])
save_csv("small_overlap.csv", ["frame a", "frame b", "candidates", "inliers", "mean error on inliers, px"], rows)
