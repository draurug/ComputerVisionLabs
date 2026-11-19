# Lab 5, part C: StereoBM and StereoSGBM, grid of parameters (about 15 minutes)

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIG, compute_disparity, load_gt, load_pair, metrics, save_csv

np.random.seed(42)

left, right = load_pair()
gt = load_gt()
print(f"ground truth: {np.mean(~np.isnan(gt)) * 100:.1f} % known, "
      f"disparity {np.nanmin(gt):.1f}...{np.nanmax(gt):.1f} px")

# Step 8. Grid of experiments
NUM_DISP = [64, 128, 192]
BLOCK = [5, 9, 15, 21]
UNIQ = [5, 10, 15]
RUNS = 3   # time = median of 3 runs

rows = []
for method in ["BM", "SGBM"]:
    for nd in NUM_DISP:
        for bs in BLOCK:
            for ur in UNIQ:
                times = []
                for _ in range(RUNS):
                    start = time.perf_counter()
                    disp = compute_disparity(method, left, right, nd, bs, ur)
                    times.append((time.perf_counter() - start) * 1000)
                m = metrics(disp, gt)
                print(f"{method:4s} nd {nd:3d} bs {bs:2d} ur {ur:2d}: coverage {m['coverage'] * 100:5.1f} %, "
                      f"bad1 {m['bad1'] * 100:5.1f} %, correct {m['correct'] * 100:5.1f} %, "
                      f"{np.median(times):6.0f} ms")
                rows.append([method, nd, bs, ur, round(m["coverage"] * 100, 2), round(m["bad1"] * 100, 2),
                             round(m["bad2"] * 100, 2), round(m["mae"], 3), round(m["correct"] * 100, 2),
                             round(np.median(times), 1)])
header = ["method", "numDisparities", "blockSize", "uniquenessRatio", "coverage, %", "bad-1, %", "bad-2, %",
          "mean abs error, px", "correct, %", "time, ms"]
save_csv("disparity_grid.csv", header, rows)

# the best combination of each method: the most ground truth pixels found within 1 px
# (coverage and accuracy together; bad-1 alone would prefer a nearly empty map)
best = {}
for method in ["BM", "SGBM"]:
    mine = [r for r in rows if r[0] == method]
    best[method] = max(mine, key=lambda r: r[8])
    print(f"best {method}: nd {best[method][1]}, bs {best[method][2]}, ur {best[method][3]}, "
          f"coverage {best[method][4]} %, bad-1 {best[method][5]} %, correct {best[method][8]} %")
save_csv("disparity_best.csv", header, [best["BM"], best["SGBM"]])

# effect of each parameter: mean over the other two, per method
effect = []
for method in ["BM", "SGBM"]:
    for col, name, values in [(1, "numDisparities", NUM_DISP), (2, "blockSize", BLOCK), (3, "uniquenessRatio", UNIQ)]:
        for v in values:
            sel = np.array([r for r in rows if r[0] == method and r[col] == v], dtype=object)
            effect.append([method, name, v] + [round(float(np.mean(sel[:, k].astype(float))), 2)
                                               for k in (4, 5, 8, 9)])
# SGBM on the color images with P1, P2 = 8 * 3 * bs^2, 32 * 3 * bs^2 (the recipe of the OpenCV
# sample): the path costs are 16-bit, and from blockSize 13 they overflow - the map becomes noise
overflow = []
for bs in [5, 9, 11, 13, 15, 21]:
    for color in [False, True]:
        m = metrics(compute_disparity("SGBM", left, right, 128, bs, 5, color=color), gt)
        overflow.append(["color" if color else "gray", bs, round(m["coverage"] * 100, 2), round(m["bad1"] * 100, 2),
                         round(m["correct"] * 100, 2)])
        print(f"SGBM {overflow[-1][0]:5s} bs {bs:2d}: coverage {m['coverage'] * 100:5.1f} %, "
              f"bad-1 {m['bad1'] * 100:5.1f} %")
save_csv("sgbm_color_vs_gray.csv", ["input", "blockSize", "coverage, %", "bad-1, %", "correct, %"], overflow)

save_csv("disparity_effect.csv", ["method", "parameter", "value", "coverage, %", "bad-1, %", "correct, %",
                                  "time, ms"], effect)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, (col, name, values) in enumerate([(1, "numDisparities", NUM_DISP), (2, "blockSize", BLOCK),
                                         (3, "uniquenessRatio", UNIQ)]):
    for method, style in [("BM", "o-"), ("SGBM", "s-")]:
        sel = [e for e in effect if e[0] == method and e[1] == name]
        ax[k].plot([e[2] for e in sel], [e[3] for e in sel], style, label=f"{method} coverage")
        ax[k].plot([e[2] for e in sel], [e[4] for e in sel], style[0] + "--", alpha=0.6,
                   label=f"{method} bad-1")
    ax[k].set_xlabel(name)
    ax[k].set_ylabel("%")
    ax[k].set_title("mean over the other two parameters")
    ax[k].grid(alpha=0.3)
ax[0].legend()
plt.savefig(os.path.join(FIG, "disparity_grid.png"), dpi=150, bbox_inches="tight")
plt.close()
