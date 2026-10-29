# Lab 3, parts A and B: own Harris vs cv2.cornerHarris, sensitivity to scale

import os

import cv2
import numpy as np

from utils import FIG, harris_points, harris_response, image_paths, matched_fraction, points_from_response, save_csv

np.random.seed(42)

# Step 8. Compare with the library function on every image.
# cornerHarris sums the products over a square blockSize window, not a gaussian, so the comparison
# is done twice: with the gaussian sigma = 1.5 from the task and with the same 5x5 box window.
print("Part A")
BLOCK = 5
rows = []
for path in image_paths():
    name = os.path.basename(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lib = points_from_response(cv2.cornerHarris(gray, blockSize=BLOCK, ksize=3, k=0.04))
    mine_gauss = harris_points(img)
    mine_box = harris_points(img, box=BLOCK)

    agree_gauss = matched_fraction(mine_gauss, lib, eps=2)
    agree_box = matched_fraction(mine_box, lib, eps=2)
    print(f"{name}: cornerHarris {len(lib)} points; gaussian window {len(mine_gauss)} points, agree {agree_gauss * 100:.1f} %;"
          f" box window {len(mine_box)} points, agree {agree_box * 100:.1f} %")
    rows.append([name, len(lib), len(mine_gauss), round(agree_gauss * 100, 1), len(mine_box), round(agree_box * 100, 1)])
save_csv("harris_vs_library.csv", ["file", "cornerHarris points", "own points, gaussian 1.5", "agree, %",
                                   "own points, box 5x5", "agree, %"], rows)

# picture: own points (gaussian) on the first image, response map next to it
img = cv2.imread(image_paths()[0])
pts = harris_points(img)
for x, y in pts:
    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), 1)
r = harris_response(cv2.imread(image_paths()[0]))
r_pic = cv2.applyColorMap((np.clip(r / r.max(), 0, 1) ** 0.3 * 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(os.path.join(FIG, "harris_points.jpg"), np.hstack([img, r_pic]))

# Steps 9-10. Half-size image, points back to the original coordinates
print("\nPart B")
rows = []
for path in image_paths():
    name = os.path.basename(path)
    img = cv2.imread(path)
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    p_full = harris_points(img)
    p_small = harris_points(small) * 2
    frac = matched_fraction(p_full, p_small)
    print(f"{name}: {len(p_full)} points on the original, {len(p_small)} on the half size, "
          f"{frac * 100:.1f} % of the original points found again")
    rows.append([name, len(p_full), len(p_small), round(frac * 100, 1)])
save_csv("harris_scale.csv", ["file", "points full size", "points half size", "matched within 3 px, %"], rows)
