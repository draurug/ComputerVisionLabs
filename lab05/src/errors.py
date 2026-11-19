# Lab 5, part E: zones of systematic errors on the best maps

import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIG, RESULTS, TMP, load_gt, load_pair, metrics, occlusion_mask, save_csv

np.random.seed(42)

TEXTURE_STD = 2.0
# polygons marked by hand on the half-size left image, (x, y)
KEYS = [(172, 485), (210, 528), (822, 357), (812, 335)]
VASE = [(1165, 400), (1190, 340), (1240, 325), (1300, 345), (1322, 410), (1300, 465), (1240, 490), (1185, 465)]
BRASS_LAMP = [(326, 75), (480, 75), (480, 105), (326, 105)]

left, _ = load_pair()
gt = load_gt()
gt1 = load_gt("disp1_gt.png")
maps = {"BM": np.load(os.path.join(TMP, "best_bm.npy")),
        "SGBM": np.load(os.path.join(TMP, "best_sgbm.npy"))}
h, w = gt.shape

# left strip x < numDisparities: the matcher has no candidates there, so it is always empty
# (the best maps of both methods use the same numDisparities, see disparity_best.csv)
nd = max(int(r["numDisparities"]) for r in csv.DictReader(open(os.path.join(RESULTS, "disparity_best.csv"))))
xs = np.arange(w)[None, :].repeat(h, axis=0)
strip = xs < nd

# occlusion: no correspondence in the right image (from the left-right check of the two ground
# truth maps); outside the strip these are the left borders of objects
occluded = occlusion_mask(gt, gt1) & ~strip

# textureless: local standard deviation of the gray image in a 15 x 15 window below 2 levels
gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
mean = cv2.blur(gray, (15, 15))
std = np.sqrt(np.maximum(cv2.blur(gray * gray, (15, 15)) - mean * mean, 0))
textureless = (std < TEXTURE_STD) & ~occluded & ~strip


def polygon(points):
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
    return mask.astype(bool) & ~occluded & ~strip


repeated = polygon(KEYS)
shiny = polygon(VASE) | polygon(BRASS_LAMP)

zones = {
    "textureless (std < 2 in 15x15)": textureless,
    "repeated pattern (piano keys)": repeated,
    "occlusion at left object borders": occluded,
    f"left strip x < {nd} (no search)": strip,
    "shiny (red vase, brass lamp)": shiny,
}
rest = ~(textureless | repeated | occluded | shiny | strip) & ~np.isnan(gt)
zones["everything else"] = rest

rows = []
for name, mask in zones.items():
    for method, disp in maps.items():
        # in the occlusions there is no true match: the right answer is a hole, any value is a guess
        m = metrics(disp, gt, mask)
        rows.append([name, method, int(mask.sum()), round(mask.mean() * 100, 2), round(m["coverage"] * 100, 2),
                     round(m["bad1"] * 100, 2), round(m["correct"] * 100, 2)])
        print(f"{name:38s} {method:4s}: {mask.sum():7d} px ({mask.mean() * 100:4.1f} %), "
              f"coverage {m['coverage'] * 100:5.1f} %, bad-1 {m['bad1'] * 100:5.1f} %, correct {m['correct'] * 100:5.1f} %")
save_csv("error_zones.csv", ["zone", "method", "pixels", "share of image, %", "coverage, %", "bad-1, %",
                             "correct, %"], rows)

# picture: zones over the image, and the SGBM error map with the same zones outlined
colors = {"textureless (std < 2 in 15x15)": (255, 255, 0), "repeated pattern (piano keys)": (255, 0, 255),
          "occlusion at left object borders": (255, 0, 0), f"left strip x < {nd} (no search)": (255, 128, 0),
          "shiny (red vase, brass lamp)": (0, 255, 0)}
overlay = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).copy()
for name, color in colors.items():
    overlay[zones[name]] = (0.4 * overlay[zones[name]] + 0.6 * np.array(color)).astype(np.uint8)

err = np.abs(maps["SGBM"] - gt)
bad = np.zeros((h, w, 3), np.uint8)
bad[~np.isnan(maps["SGBM"])] = (60, 60, 60)
bad[err > 1] = (255, 255, 255)
for name, color in colors.items():
    contours, _ = cv2.findContours(zones[name].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bad, [c for c in contours if cv2.contourArea(c) > 30], -1, color, 2)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(overlay)
ax[0].set_title("zones")
ax[1].imshow(bad)
ax[1].set_title("SGBM: white = |d - d_gt| > 1 px, gray = correct, black = no value")
for a in ax:
    a.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in colors.values()]
fig.legend(handles, list(colors), loc="lower center", ncol=5, fontsize=9)
plt.savefig(os.path.join(FIG, "error_zones.png"), dpi=150, bbox_inches="tight")
plt.close()
