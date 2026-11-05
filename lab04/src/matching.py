# Lab 4, parts A and B: ratio test, mutual check, least-squares homography without outlier removal

import os

import cv2
import numpy as np

from utils import FIG, features, load, match_points, mutual_matches, ratio_matches, reprojection_error, save_csv

np.random.seed(42)

NAME1, NAME2 = "rot_03.jpg", "rot_04.jpg"   # neighbouring frames, about 40 % overlap
img1, img2 = load(NAME1), load(NAME2)
kp1, desc1 = features(img1)
kp2, desc2 = features(img2)
print(f"{NAME1}: {len(kp1)} points, {NAME2}: {len(kp2)} points")

# Steps 3-4. Candidates for different ratio thresholds, with and without the mutual check
rows = []
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    ratio = ratio_matches(desc1, desc2, t)
    mutual = mutual_matches(desc1, desc2, t)
    print(f"T = {t}: ratio test {len(ratio)}, + mutual {len(mutual)}")
    rows.append([t, len(ratio), len(mutual)])
save_csv("ratio_test.csv", ["T", "ratio test", "ratio test + mutual"], rows)

# Step 5. Raw candidates at T = 0.8, wrong lines are visible
raw = ratio_matches(desc1, desc2, 0.8)
pic = cv2.drawMatches(img1, kp1, img2, kp2, raw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(FIG, "raw_matches.jpg"), pic)

# Steps 6-7. Plain least squares on all candidates (method 0 = no outlier rejection)
src, dst = match_points(kp1, kp2, raw)
H, _ = cv2.findHomography(src, dst, 0)
err = reprojection_error(H, src, dst)
print(f"\nleast squares on {len(raw)} candidates: mean error {err.mean():.1f} px, max {err.max():.1f} px")

h, w = img2.shape[:2]
warped = cv2.warpPerspective(img1, H, (w * 2, h))
cv2.imwrite(os.path.join(FIG, "lsq_warp.jpg"), warped)
save_csv("lsq_homography.csv", ["candidates", "mean error, px", "median error, px", "max error, px"],
         [[len(raw), round(err.mean(), 2), round(np.median(err), 2), round(err.max(), 2)]])
