# Lab 2, step 12: how the error and fx depend on the number of frames

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FIG, RESULTS, object_points, per_frame_errors, save_csv

np.random.seed(42)

saved = np.load(os.path.join(RESULTS, "corners.npz"))
corners = saved["corners"]
image_size = tuple(int(v) for v in saved["image_size"])
total = len(corners)
obj = object_points()

rows = []
summary = []
for n in [5, 8, 10, 12, 15, 20]:
    errs = []
    fxs = []
    for trial in range(5):
        idx = np.random.choice(total, n, replace=False)
        pts = [corners[i] for i in idx]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * n, pts, image_size, None, None)
        e = per_frame_errors([obj] * n, pts, K, dist, rvecs, tvecs).mean()
        errs.append(e)
        fxs.append(K[0, 0])
        rows.append([n, trial + 1, round(e, 4), round(K[0, 0], 1), round(K[1, 1], 1)])
    print(f"N = {n:2d}: error {np.mean(errs):.3f} ± {np.std(errs):.3f} px, fx {np.mean(fxs):.1f} ± {np.std(fxs):.1f}")
    summary.append([n, np.mean(errs), np.std(errs), np.mean(fxs), np.std(fxs)])

save_csv("frames_vs_n.csv", ["N", "trial", "mean error, px", "fx", "fy"], rows)
save_csv("frames_vs_n_summary.csv", ["N", "error mean, px", "error std", "fx mean", "fx std"],
         [[s[0]] + [round(v, 3) for v in s[1:]] for s in summary])

summary = np.array(summary)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(summary[:, 0], summary[:, 1], summary[:, 2], fmt="o-", capsize=4)
ax[0].set_xlabel("number of frames N")
ax[0].set_ylabel("mean reprojection error, px")
ax[1].errorbar(summary[:, 0], summary[:, 3], summary[:, 4], fmt="o-", capsize=4, color="tab:orange")
ax[1].set_xlabel("number of frames N")
ax[1].set_ylabel("fx, px")
for a in ax:
    a.grid(alpha=0.3)
plt.savefig(os.path.join(FIG, "frames_vs_n.png"), dpi=150, bbox_inches="tight")
plt.close()
