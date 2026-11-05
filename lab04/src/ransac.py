# Lab 4, part C: own RANSAC for the homography, comparison with cv2, threshold study

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIG, features, load, match_points, ratio_matches, reprojection_error, save_csv

np.random.seed(42)

NAME1, NAME2 = "rot_03.jpg", "rot_04.jpg"
img1, img2 = load(NAME1), load(NAME2)
kp1, desc1 = features(img1)
kp2, desc2 = features(img2)
matches = ratio_matches(desc1, desc2, 0.8)
src, dst = match_points(kp1, kp2, matches)
print(f"{len(matches)} candidates")


def ransac(src, dst, tau, iterations=1000):
    # Step 8: random 4 pairs -> H -> count the pairs that agree within tau, keep the best,
    # then refit H on all the inliers with plain least squares
    best_inliers = np.zeros(len(src), dtype=bool)
    for _ in range(iterations):
        idx = np.random.choice(len(src), 4, replace=False)
        H = cv2.getPerspectiveTransform(src[idx], dst[idx])
        inliers = reprojection_error(H, src, dst) < tau
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    H, _ = cv2.findHomography(src[best_inliers], dst[best_inliers], 0)
    return H, best_inliers


# Step 9. Needed number of iterations from the observed outlier share
TAU = 3
ITERATIONS = 1000
H_own, inl = ransac(src, dst, TAU, ITERATIONS)
outlier_share = 1 - inl.mean()
needed = np.log(1 - 0.99) / np.log(1 - (1 - outlier_share) ** 4)
print(f"\ntau = {TAU}: {inl.sum()} inliers of {len(src)}, outlier share {outlier_share:.2f}, "
      f"needed iterations for p = 0.99: {needed:.0f} (used {ITERATIONS})")

# Step 10. Compare with cv2.findHomography RANSAC
H_cv, mask = cv2.findHomography(src, dst, cv2.RANSAC, TAU)
inl_cv = mask.ravel().astype(bool)
err_own = reprojection_error(H_own, src, dst)[inl]
err_cv = reprojection_error(H_cv, src, dst)[inl_cv]
print(f"own RANSAC: {inl.sum()} inliers, error on them {err_own.mean():.3f} px")
print(f"cv2 RANSAC: {inl_cv.sum()} inliers, error on them {err_cv.mean():.3f} px")
print(f"same inliers: {np.sum(inl & inl_cv)}")
save_csv("ransac_vs_cv2.csv", ["method", "inliers", "mean error on inliers, px", "iterations"], [
    ["own", inl.sum(), round(err_own.mean(), 3), ITERATIONS],
    ["cv2.findHomography RANSAC", inl_cv.sum(), round(err_cv.mean(), 3), ""],
    ["needed for p = 0.99", "", "", round(needed)],
])

# Step 11. Threshold study
rows = []
taus = [1, 2, 3, 5, 8, 12, 20]
counts = []
errors = []
for tau in taus:
    H, inliers = ransac(src, dst, tau, ITERATIONS)
    e = reprojection_error(H, src, dst)[inliers]
    counts.append(inliers.sum())
    errors.append(e.mean())
    print(f"tau = {tau:2d}: {inliers.sum()} inliers, error on them {e.mean():.2f} px")
    rows.append([tau, inliers.sum(), round(e.mean(), 3)])
save_csv("ransac_threshold.csv", ["tau, px", "inliers", "mean error on inliers, px"], rows)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(taus, counts, "o-")
ax[0].set_xlabel("tau, px")
ax[0].set_ylabel("inliers")
ax[1].plot(taus, errors, "o-", color="tab:orange")
ax[1].set_xlabel("tau, px")
ax[1].set_ylabel("mean reprojection error on inliers, px")
for a in ax:
    a.grid(alpha=0.3)
plt.savefig(os.path.join(FIG, "ransac_threshold.png"), dpi=150, bbox_inches="tight")
plt.close()

# Step 12. Inliers green, outliers red, on one picture
pic = np.hstack([img1, img2])
shift = img1.shape[1]
for (x1, y1), (x2, y2), ok in zip(src, dst, inl):
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.line(pic, (int(x1), int(y1)), (int(x2) + shift, int(y2)), color, 1)
cv2.imwrite(os.path.join(FIG, "ransac_matches.jpg"), pic)
