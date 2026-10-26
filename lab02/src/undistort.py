# Lab 2, steps 13-14: undistort with alpha = 0 and 1, check that straight lines become straight

import os

import cv2
import numpy as np

from utils import BOARD, DATA, FIG, RESULTS, save_csv

fs = cv2.FileStorage(os.path.join(RESULTS, "camera_params.yml"), cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("dist_coeffs").mat()
fs.release()

saved = np.load(os.path.join(RESULTS, "corners.npz"))
names = list(saved["names"])
corners = saved["corners"]
w, h = [int(v) for v in saved["image_size"]]

# Step 13. alpha = 0 crops the frame, alpha = 1 keeps everything and leaves black borders
name = "calib_01.jpg"   # board at the top edge, table edge visible
img = cv2.imread(os.path.join(DATA, name))
out = [img]
for alpha in [0, 1]:
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    fixed = cv2.undistort(img, K, dist, None, newK)
    print(f"alpha = {alpha}: valid roi (x, y, w, h) = {roi}, fx = {newK[0, 0]:.1f}")
    x, y, rw, rh = roi
    cv2.rectangle(fixed, (x, y), (x + rw, y + rh), (0, 0, 255), 6)
    out.append(fixed)
small = [cv2.resize(o, None, fx=0.3, fy=0.3) for o in out]
cv2.imwrite(os.path.join(FIG, "undistort_alpha.jpg"), np.hstack(small))

# Step 14. Straight lines. There is no separate frame with a doorway, so the rows of the board
# are used: a line through the first and last corner of a row, deviation of the corners between them.
# Before undistortion the corners are the detected ones, after - the same corners mapped with undistortPoints.
rows = []
worst_before = 0
for name, pts in zip(names, corners):
    pts = pts.reshape(BOARD[1], BOARD[0], 2)
    fixed = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(BOARD[1], BOARD[0], 2)
    dev = []
    for grid in [pts, fixed]:
        d = []
        for row in grid:
            a, b = row[0], row[-1]
            direction = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-direction[1], direction[0]])
            d.append(np.abs((row[1:-1] - a) @ normal).max())
        dev.append(max(d))
    rows.append([name, round(dev[0], 3), round(dev[1], 3)])
    if dev[0] > worst_before:
        worst_before = dev[0]
        worst_name = name

save_csv("straight_lines.csv", ["file", "max deviation before, px", "max deviation after, px"], rows)
arr = np.array([[r[1], r[2]] for r in rows])
print(f"line deviation: before {arr[:, 0].mean():.2f} px (max {arr[:, 0].max():.2f}), "
      f"after {arr[:, 1].mean():.2f} px (max {arr[:, 1].max():.2f})")

# picture: the frame with the largest deviation, lines drawn through the end corners of each row
print("worst frame:", worst_name)
img = cv2.imread(os.path.join(DATA, worst_name))
pts = corners[names.index(worst_name)].reshape(BOARD[1], BOARD[0], 2)
fixed_img = cv2.undistort(img, K, dist)
fixed_pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(BOARD[1], BOARD[0], 2)


def draw(image, grid):
    for row in grid:
        a, b = row[0], row[-1]
        # extend the line past the board
        d = (b - a) / np.linalg.norm(b - a) * 400
        cv2.line(image, tuple((a - d).astype(int)), tuple((b + d).astype(int)), (0, 0, 255), 2)
        for p in row:
            cv2.circle(image, tuple(p.astype(int)), 8, (0, 255, 0), 2)
    return image


before = draw(img.copy(), pts)
after = draw(fixed_img.copy(), fixed_pts)
both = np.hstack([before, after])
cv2.imwrite(os.path.join(FIG, "straight_lines.jpg"), cv2.resize(both, None, fx=0.4, fy=0.4))

# crop of the same, to actually see the deviation
y0, x0 = pts[:, :, 1].min().astype(int), pts[:, :, 0].min().astype(int)
y1, x1 = pts[:, :, 1].max().astype(int), pts[:, :, 0].max().astype(int)
crop = np.hstack([before[y0 - 40:y1 + 40, x0 - 40:x1 + 40], after[y0 - 40:y1 + 40, x0 - 40:x1 + 40]])
cv2.imwrite(os.path.join(FIG, "straight_lines_crop.jpg"), crop)
