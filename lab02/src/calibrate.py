# Lab 2, steps 6-11 and 15: corners, calibration, per-frame errors, outliers, camera_params.yml

import os
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import BOARD, FIG, RESULTS, SQUARE, find_corners, image_paths, object_points, per_frame_errors, refine, save_csv

np.random.seed(42)

# Step 6. Find corners on every frame
print("Step 6")
names = []
raw_corners = []      # as returned by the detector
sub_corners = []      # after cornerSubPix
detect_rows = []
image_size = None

for path in image_paths():
    name = os.path.basename(path)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_size = (gray.shape[1], gray.shape[0])
    corners, how = find_corners(gray)
    print(f"{name}: {how}")
    detect_rows.append([name, how])
    if corners is None:
        continue
    names.append(name)
    raw_corners.append(corners.astype(np.float32))
    sub_corners.append(refine(gray, corners))

save_csv("detection.csv", ["file", "detector"], detect_rows)
print(f"found on {len(names)} of {len(detect_rows)} frames")

# example of detected corners
img = cv2.imread(os.path.join(os.path.dirname(image_paths()[0]), names[0]))
cv2.drawChessboardCorners(img, BOARD, sub_corners[0], True)
cv2.imwrite(os.path.join(FIG, "corners_example.jpg"), cv2.resize(img, None, fx=0.5, fy=0.5))

# Step 8. 3D points of the board, the same for every frame
objpoints = [object_points()] * len(names)

# Step 7 and 9. Calibrate with and without subpixel refinement
print("\nSteps 7, 9")
compare_rows = []
for label, pts in [("without cornerSubPix", raw_corners), ("with cornerSubPix", sub_corners)]:
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, pts, image_size, None, None)
    err = per_frame_errors(objpoints, pts, K, dist, rvecs, tvecs)
    print(f"{label}: rms = {rms:.4f}, mean per-frame error = {err.mean():.4f} px, fx = {K[0, 0]:.1f}")
    compare_rows.append([label, round(rms, 4), round(err.mean(), 4), round(K[0, 0], 1), round(K[1, 1], 1)])
save_csv("subpix_effect.csv", ["corners", "rms from calibrateCamera", "mean per-frame error, px", "fx", "fy"],
         compare_rows)

# from here on only the refined corners are used
imgpoints = sub_corners

# Step 10. Error per frame (own computation with projectPoints)
print("\nStep 10")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
errors = per_frame_errors(objpoints, imgpoints, K, dist, rvecs, tvecs)
median = np.median(errors)
for name, e in zip(names, errors):
    print(f"{name}: {e:.3f} px")
print(f"median = {median:.3f} px")

# Step 11. Drop frames with error > 2 * median, calibrate again
threshold = 2 * median
keep = errors <= threshold
dropped = [n for n, k in zip(names, keep) if not k]
print(f"\nStep 11: threshold = {threshold:.3f} px, dropped {len(dropped)} frames: {dropped}")

names2 = [n for n, k in zip(names, keep) if k]
objpoints2 = [o for o, k in zip(objpoints, keep) if k]
imgpoints2 = [p for p, k in zip(imgpoints, keep) if k]
rms2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, image_size, None, None)
errors2 = per_frame_errors(objpoints2, imgpoints2, K2, dist2, rvecs2, tvecs2)

save_csv("frame_errors.csv", ["file", "error before, px", "kept", "error after, px"],
         [[n, round(e, 4), int(k), round(errors2[names2.index(n)], 4) if k else ""]
          for n, e, k in zip(names, errors, keep)])

plt.figure(figsize=(10, 4))
plt.bar(range(len(names)), errors, color=["tab:blue" if k else "tab:red" for k in keep])
plt.axhline(median, color="gray", linestyle="--", label=f"median = {median:.3f}")
plt.axhline(threshold, color="red", linestyle="--", label=f"2 × median = {threshold:.3f}")
plt.xticks(range(len(names)), [n[6:8] for n in names])
plt.xlabel("frame")
plt.ylabel("mean reprojection error, px")
plt.legend()
plt.savefig(os.path.join(FIG, "frame_errors.png"), dpi=150, bbox_inches="tight")
plt.close()

# final numbers
w, h = image_size
print("\nFinal calibration")
print("K =\n", K2.round(2))
print("dist =", dist2.ravel().round(5))
print(f"rms = {rms2:.4f}, mean per-frame error = {errors2.mean():.4f} px")
print(f"fx / fy = {K2[0, 0] / K2[1, 1]:.4f}")
print(f"principal point: ({K2[0, 2]:.1f}, {K2[1, 2]:.1f}), image center: ({w / 2}, {h / 2}), "
      f"offset {(K2[0, 2] - w / 2) / w * 100:.1f} % x, {(K2[1, 2] - h / 2) / h * 100:.1f} % y")

save_csv("calibration.csv", ["parameter", "all frames", "after dropping outliers"], [
    ["frames", len(names), len(names2)],
    ["rms (calibrateCamera)", round(rms, 4), round(rms2, 4)],
    ["mean per-frame error, px", round(errors.mean(), 4), round(errors2.mean(), 4)],
    ["fx", round(K[0, 0], 2), round(K2[0, 0], 2)],
    ["fy", round(K[1, 1], 2), round(K2[1, 1], 2)],
    ["fx / fy", round(K[0, 0] / K[1, 1], 4), round(K2[0, 0] / K2[1, 1], 4)],
    ["cx", round(K[0, 2], 2), round(K2[0, 2], 2)],
    ["cy", round(K[1, 2], 2), round(K2[1, 2], 2)],
    ["k1", round(dist[0, 0], 5), round(dist2[0, 0], 5)],
    ["k2", round(dist[0, 1], 5), round(dist2[0, 1], 5)],
    ["p1", round(dist[0, 2], 5), round(dist2[0, 2], 5)],
    ["p2", round(dist[0, 3], 5), round(dist2[0, 3], 5)],
    ["k3", round(dist[0, 4], 5), round(dist2[0, 4], 5)],
])

# Step 15. Save
fs = cv2.FileStorage(os.path.join(RESULTS, "camera_params.yml"), cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K2)
fs.write("dist_coeffs", dist2)
fs.write("image_width", w)
fs.write("image_height", h)
fs.write("square_size_mm", SQUARE)
fs.write("reprojection_error_px", float(errors2.mean()))
fs.write("frames_used", len(names2))
fs.write("date", datetime.now().strftime("%Y-%m-%d"))
fs.release()
print("saved camera_params.yml")

# corners of the kept frames, for the other scripts
np.savez(os.path.join(RESULTS, "corners.npz"), names=names2, corners=np.array(imgpoints2), image_size=image_size)
