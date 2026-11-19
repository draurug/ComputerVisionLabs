# Lab 5, part C, step 9: the best maps of BM and SGBM (from disparity_best.csv) with a color scale

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIG, RESULTS, TMP, compute_disparity, load_gt, load_pair

np.random.seed(42)

left, right = load_pair()
gt = load_gt()
best = {r["method"]: r for r in csv.DictReader(open(os.path.join(RESULTS, "disparity_best.csv")))}
params = {m: (int(best[m]["numDisparities"]), int(best[m]["blockSize"]), int(best[m]["uniquenessRatio"]))
          for m in ["BM", "SGBM"]}
maps = {m: compute_disparity(m, left, right, *params[m]) for m in ["BM", "SGBM"]}
np.save(os.path.join(TMP, "best_bm.npy"), maps["BM"])
np.save(os.path.join(TMP, "best_sgbm.npy"), maps["SGBM"])

turbo = plt.get_cmap("turbo").copy()
turbo.set_bad("black")
magma = plt.get_cmap("magma").copy()
magma.set_bad("black")
vmax = np.nanmax(gt)
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
panels = [("ground truth", gt)] + [(f"Stereo{m} nd={params[m][0]} bs={params[m][1]} ur={params[m][2]}", maps[m])
                                   for m in ["BM", "SGBM"]]
for a, (title, d) in zip(ax.flat[:3], panels):
    im = a.imshow(d, cmap=turbo, vmin=0, vmax=vmax)
    a.set_title(title + " (black = no value)")
    a.axis("off")
    plt.colorbar(im, ax=a, fraction=0.03, label="disparity, px")
im = ax[1, 1].imshow(np.abs(maps["SGBM"] - gt), cmap=magma, vmin=0, vmax=5)
ax[1, 1].set_title("SGBM |d - d_gt| (black = no value)")
ax[1, 1].axis("off")
plt.colorbar(im, ax=ax[1, 1], fraction=0.03, label="error, px")
plt.savefig(os.path.join(FIG, "disparity_best.png"), dpi=150, bbox_inches="tight")
plt.close()
