# Lab 5, part D: depth from disparity, check against the ground truth, point cloud

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils import FIG, TMP, depth_from_disparity, load_gt, load_pair, save_csv, scaled_calib

np.random.seed(42)

cal = scaled_calib()
left, _ = load_pair()
gt = load_gt()
disp = np.load(os.path.join(TMP, "best_sgbm.npy"))   # made by best_map.py
Q = np.load(os.path.join(TMP, "Q.npy"))              # made by rectify.py

# Step 10. Z = f * B / (d + doffs), and the same through reprojectImageTo3D with Q
Z = depth_from_disparity(disp, cal)
Z_gt = depth_from_disparity(gt, cal)
xyz = cv2.reprojectImageTo3D(np.nan_to_num(disp, nan=-1).astype(np.float32), Q) / 1000.0   # mm -> m
both = ~np.isnan(disp)
print(f"Z by formula vs reprojectImageTo3D: max difference {np.nanmax(np.abs(Z[both] - xyz[..., 2][both])):.2e} m")
print(f"ground truth depth {np.nanmin(Z_gt):.2f}...{np.nanmax(Z_gt):.2f} m")

# Step 11. Instead of a tape measure (variant 3): the ground truth depth of 7 objects at different
# distances. Each object is a 15 x 15 window, the median of the window is taken for both maps.
OBJECTS = [
    # name, x, y in the half-size image
    ("piano front under the keys", 250, 560),
    ("bench seat", 560, 600),
    ("sheet music", 480, 260),
    ("wall right of guitar", 1060, 250),
    ("guitar body", 930, 650),
    ("black folder", 600, 150),
    ("wall under ceiling", 750, 150),
]
WIN = 7
rows = []
for name, x, y in OBJECTS:
    z_gt = np.nanmedian(Z_gt[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    z = np.nanmedian(Z[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    d_gt = np.nanmedian(gt[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    d = np.nanmedian(disp[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1])
    rows.append([name, x, y, round(d_gt, 2), round(d, 2), round(z_gt, 3), round(z, 3), round(abs(z - z_gt) * 1000, 1),
                 round(abs(z - z_gt) / z_gt * 100, 2)])
    print(f"{name:20s} gt {z_gt:.3f} m, computed {z:.3f} m, error {abs(z - z_gt) * 1000:6.1f} mm "
          f"({abs(z - z_gt) / z_gt * 100:.2f} %)")
rows.sort(key=lambda r: r[5])
save_csv("depth_objects.csv", ["object", "x", "y", "gt disparity, px", "disparity, px", "gt depth, m",
                               "computed depth, m", "abs error, mm", "rel error, %"], rows)

# The same over the whole image: error vs ground truth depth in bins. One disparity step dd gives
# dZ = Z^2 / (f B) * dd, so the absolute error should grow as Z^2 and the relative one as Z.
ok = ~np.isnan(Z) & ~np.isnan(Z_gt) & (np.abs(disp - gt) <= 3)   # without gross mismatches
edges = np.quantile(Z_gt[ok], np.linspace(0, 1, 9))
centers, abs_err, rel_err, theory = [], [], [], []
bin_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = ok & (Z_gt >= lo) & (Z_gt < hi)
    e = np.abs(Z[sel] - Z_gt[sel])
    zc = Z_gt[sel].mean()
    dd = np.abs(disp[sel] - gt[sel]).mean()
    centers.append(zc)
    abs_err.append(e.mean() * 1000)
    rel_err.append((e / Z_gt[sel]).mean() * 100)
    theory.append(zc ** 2 / (cal["f"] * cal["B"] / 1000) * dd * 1000)
    bin_rows.append([round(lo, 3), round(hi, 3), sel.sum(), round(dd, 3), round(abs_err[-1], 1),
                     round(rel_err[-1], 3), round(theory[-1], 1)])
save_csv("depth_error_bins.csv", ["Z from, m", "Z to, m", "pixels", "mean |dd|, px", "mean abs error, mm",
                                  "mean rel error, %", "Z^2/(fB)*|dd|, mm"], bin_rows)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(centers, abs_err, "o-", label="measured")
ax[0].plot(centers, theory, "--", label="Z²/(f·B)·mean|Δd|")
ax[0].set_xlabel("ground truth depth, m")
ax[0].set_ylabel("mean absolute depth error, mm")
ax[0].legend()
ax[1].plot(centers, rel_err, "o-", color="tab:orange", label="measured")
# with the same disparity error everywhere the relative error is Z / (f B) * |dd|, a straight line
dd_all = np.abs(disp[ok] - gt[ok]).mean()
zz = np.linspace(min(centers), max(centers), 20)
ax[1].plot(zz, zz / (cal["f"] * cal["B"] / 1000) * dd_all * 100, "--", color="gray",
           label=f"Z/(f·B)·{dd_all:.2f} px (mean |Δd| over all bins)")
ax[1].legend()
ax[1].set_xlabel("ground truth depth, m")
ax[1].set_ylabel("mean relative depth error, %")
for a in ax:
    a.grid(alpha=0.3)
plt.savefig(os.path.join(FIG, "depth_error.png"), dpi=150, bbox_inches="tight")
plt.close()

# depth map with the objects
fig, ax = plt.subplots(figsize=(9, 6))
cmap = plt.get_cmap("turbo_r").copy()
cmap.set_bad("black")
im = ax.imshow(Z, cmap=cmap, vmin=np.nanpercentile(Z_gt, 1), vmax=np.nanpercentile(Z_gt, 99))
ax.set_title("depth from the SGBM map (black = no value)")
for name, x, y, *_ in rows:
    ax.plot(x, y, "w+", markersize=12, mew=2)
    ax.text(x + 8, y - 8, name, color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1))
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.03, label="depth Z, m")
plt.savefig(os.path.join(FIG, "depth_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# Step 12. Point cloud: drop invalid pixels, keep the depth range of the scene, remove isolated
# points (mean distance to the 8 nearest neighbours above mean + 2 std)
Z_MIN, Z_MAX = np.nanpercentile(Z_gt, 0.5), np.nanpercentile(Z_gt, 99.5)
valid = ~np.isnan(disp)
in_range = valid & (xyz[..., 2] > Z_MIN) & (xyz[..., 2] < Z_MAX)
pts = xyz[in_range]
colors = left[in_range][:, ::-1]   # BGR -> RGB
dist, _ = cKDTree(pts).query(pts, k=9)
mean_d = dist[:, 1:].mean(axis=1)
keep = mean_d < mean_d.mean() + 2 * mean_d.std()
pts, colors = pts[keep], colors[keep]
n_all = left.shape[0] * left.shape[1]
print(f"point cloud: {n_all} pixels, {valid.sum()} valid, {in_range.sum()} in {Z_MIN:.2f}...{Z_MAX:.2f} m, "
      f"{len(pts)} after removing isolated points")
save_csv("point_cloud.csv", ["stage", "points"], [
    ["all pixels", n_all], ["valid disparity", int(valid.sum())],
    [f"depth {Z_MIN:.2f}...{Z_MAX:.2f} m", int(in_range.sum())], ["isolated removed", len(pts)]])

with open(os.path.join(TMP, "cloud.ply"), "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(pts, colors):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

sub = np.random.choice(len(pts), min(40000, len(pts)), replace=False)
fig = plt.figure(figsize=(13, 5))
a = fig.add_subplot(1, 2, 1, projection="3d")
a.scatter(pts[sub, 0], pts[sub, 2], -pts[sub, 1], c=colors[sub] / 255, s=0.3)
a.set_xlabel("X, m")
a.set_ylabel("Z, m")
a.set_zlabel("-Y, m")
a.view_init(elev=20, azim=-65)
a.set_title("point cloud")
a = fig.add_subplot(1, 2, 2)
a.scatter(pts[sub, 0], pts[sub, 2], c=colors[sub] / 255, s=0.3)
a.set_xlabel("X, m")
a.set_ylabel("Z (depth), m")
a.set_aspect("equal")
a.grid(alpha=0.3)
a.set_title("view from above")
plt.savefig(os.path.join(FIG, "point_cloud.png"), dpi=150, bbox_inches="tight")
plt.close()
