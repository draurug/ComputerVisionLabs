# Lab 7, parts B and C: Otsu, watershed, SLIC, GrabCut on the 10 own images, IoU and Dice against the
# own masks; the pixel accuracy trap on a small object

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries, slic

from utils import FIG, NAMES, border_fraction, dice, fit, hstack, iou, label, load_image, load_mask, overlay, \
    pixel_accuracy, save_csv, vstack

np.random.seed(42)
cv2.setRNGSeed(42)

METHODS = ["Otsu", "watershed", "SLIC", "GrabCut"]

# None of the methods knows which of the two classes is the object. The same rule is used for all of them
# (no tuning on the masks): the class that covers less of the image border is the object.


def orient(mask):
    return ~mask if border_fraction(mask) > 0.5 else mask


def otsu(image):
    # Step 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return orient(binary > 0)


def watershed(image):
    # Step 4, the sequence from the task
    binary = otsu(image).astype(np.uint8) * 255                                   # 1. Otsu
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)       # 2. opening
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)                         # 3. distance transform
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)   # 4. 0.5 of max = sure object
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = 255 - cv2.dilate(opened, kernel, iterations=3)                     # 5. dilate + invert = sure bg
    n, markers = cv2.connectedComponents(sure_fg)                                # 6. markers
    markers = markers + 1                    # object markers 2..n, the sure background gets label 1
    markers[(sure_fg == 0) & (sure_bg == 0)] = 0     # unknown zone, watershed decides
    markers[sure_bg > 0] = 1
    markers = cv2.watershed(image, markers)                                      # 7. watershed
    return markers > 1                       # -1 = borders between regions, 1 = background


def superpixels(image):
    return slic(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), n_segments=400, compactness=10, start_label=0)


def slic_two_classes(image, segments=None):
    # Step 5: SLIC, then the superpixels are split into two classes by their mean colour (k-means, k = 2)
    if segments is None:
        segments = superpixels(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    flat = segments.ravel()
    n = flat.max() + 1
    counts = np.bincount(flat, minlength=n)
    means = np.stack([np.bincount(flat, lab[:, c], minlength=n) for c in range(3)], 1) / np.maximum(counts, 1)[:, None]
    means = means[counts > 0].astype(np.float32)
    ids = np.nonzero(counts > 0)[0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, _ = cv2.kmeans(means, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    cls = np.zeros(n, bool)
    cls[ids] = labels.ravel() == 1
    return orient(cls[segments])


def grabcut_rect(gt, margin=0.05):
    # the rectangle a user would draw: bounding box of the object plus 5 % of its size on every side
    ys, xs = np.nonzero(gt)
    h, w = gt.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0 = max(int(x0 - dx), 0), max(int(y0 - dy), 0)
    x1, y1 = min(int(x1 + dx), w - 1), min(int(y1 + dy), h - 1)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def grabcut(image, rect):
    # Step 6: rectangle around the object, 5 iterations
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


rows = []
results = {}
for name in NAMES:
    image = load_image(name)
    gt = load_mask(name)
    segments = superpixels(image)
    rect = grabcut_rect(gt)
    preds = {"Otsu": otsu(image), "watershed": watershed(image), "SLIC": slic_two_classes(image, segments),
             "GrabCut": grabcut(image, rect)}
    results[name] = (image, gt, preds, segments, rect)
    line = []
    for m in METHODS:
        p = preds[m]
        rows.append([name, m, round(iou(p, gt), 4), round(dice(p, gt), 4), round(pixel_accuracy(p, gt), 4),
                     round(gt.mean(), 4), round(p.mean(), 4)])
        line.append(f"{m} {iou(p, gt):.3f}")
    print(f"{name:13s} object {gt.mean() * 100:4.1f} %  IoU: " + ", ".join(line))

save_csv("segmentation.csv", ["image", "method", "IoU", "Dice", "pixel accuracy", "object share",
                              "predicted share"], rows)

# Step 7. table: method - mean IoU - std - min - max (and the same for Dice), over all 10 images and over
# the 6 images with a large object and the 4 houses separately
summary = []
groups = {"all 10": NAMES, "large objects (6)": NAMES[:6], "houses (4)": NAMES[6:]}
for group, names in groups.items():
    for m in METHODS:
        ious = np.array([r[2] for r in rows if r[1] == m and r[0] in names])
        dices = np.array([r[3] for r in rows if r[1] == m and r[0] in names])
        accs = np.array([r[4] for r in rows if r[1] == m and r[0] in names])
        summary.append([group, m, round(ious.mean(), 3), round(ious.std(ddof=1), 3), round(ious.min(), 3),
                        round(ious.max(), 3), round(np.median(ious), 3), round(dices.mean(), 3),
                        round(dices.std(ddof=1), 3), round(accs.mean(), 3)])
        if group == "all 10":
            print(f"{m:10s} IoU {ious.mean():.3f} +- {ious.std(ddof=1):.3f} [{ious.min():.3f}, {ious.max():.3f}], "
                  f"Dice {dices.mean():.3f} +- {dices.std(ddof=1):.3f}, accuracy {accs.mean():.3f}")
save_csv("segmentation_summary.csv", ["images", "method", "IoU mean", "IoU std", "IoU min", "IoU max",
                                      "IoU median", "Dice mean", "Dice std", "accuracy mean"], summary)

# plot: IoU of every image for every method, so the spread is visible
plt.figure(figsize=(7.5, 4))
markers = {"honey": "o", "hoop": "D", "sheet": "s", "house": "^"}
colors = plt.cm.tab10(np.arange(len(NAMES)))
for i, m in enumerate(METHODS):
    for j, name in enumerate(NAMES):
        v = [r[2] for r in rows if r[0] == name and r[1] == m][0]
        plt.scatter(i + (j - 4.5) * 0.05, v, color=colors[j], marker=markers[name.split("_")[0]], s=36,
                    label=name if i == 0 else None, zorder=3)
    ious = [r[2] for r in rows if r[1] == m]
    plt.plot([i - 0.3, i + 0.3], [np.mean(ious)] * 2, "k-", lw=2, zorder=2)
plt.xticks(range(len(METHODS)), METHODS)
plt.ylabel("IoU with the own mask")
plt.ylim(-0.03, 1.03)
plt.title("IoU per image (black line = mean over 10 images)")
plt.grid(axis="y", alpha=0.3)
plt.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.savefig(os.path.join(FIG, "iou_per_image.png"), dpi=150, bbox_inches="tight")
plt.close()

# picture: image with the own mask, then the four methods, for a few images
picture = []
for name in ["honey_lamp", "hoop", "sheet_01", "house_0663"]:
    image, gt, preds, segments, rect = results[name]
    scores = {m: iou(preds[m], gt) for m in METHODS}     # on the whole image, also for the cropped house
    if name.startswith("house"):
        # the house is small, show the part of the image around it
        image, gt = image[180:480, 0:400], gt[180:480, 0:400]
        preds = {m: p[180:480, 0:400] for m, p in preds.items()}
    first = overlay(image, gt, (0, 200, 0))
    tiles = [label(fit(first, 260), "own mask", 0.6)]
    for m in METHODS:
        tile = overlay(image, preds[m], (0, 0, 255))
        if m == "GrabCut" and not name.startswith("house"):
            x, y, w, h = rect
            cv2.rectangle(tile, (x, y), (x + w, y + h), (255, 255, 0), 3)
        tiles.append(label(fit(tile, 260), f"{m}: IoU {scores[m]:.2f}", 0.6))
    picture.append(hstack(tiles))
cv2.imwrite(os.path.join(FIG, "segmentation_examples.jpg"), vstack(picture), [cv2.IMWRITE_JPEG_QUALITY, 88])

# SLIC superpixels on one image, as the preprocessing step
image, gt, preds, segments, rect = results["honey_lamp"]
sp = (mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments, color=(1, 1, 0)) * 255).astype(np.uint8)
cv2.imwrite(os.path.join(FIG, "slic_superpixels.jpg"),
            hstack([fit(cv2.cvtColor(sp, cv2.COLOR_RGB2BGR), 420),
                    fit(overlay(image, preds["SLIC"], (0, 0, 255)), 420)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

# Part C, step 8. object < 10 % of the image: the empty mask (all background) against IoU
trap = []
for name in NAMES[6:]:
    image, gt, preds, segments, rect = results[name]
    empty = np.zeros_like(gt)
    trap.append([name, round(gt.mean(), 4), "empty mask", round(pixel_accuracy(empty, gt), 4), round(iou(empty, gt), 4),
                 round(dice(empty, gt), 4)])
    for m in METHODS:
        trap.append([name, round(gt.mean(), 4), m, round(pixel_accuracy(preds[m], gt), 4), round(iou(preds[m], gt), 4),
                     round(dice(preds[m], gt), 4)])
save_csv("accuracy_trap.csv", ["image", "object share", "prediction", "pixel accuracy", "IoU", "Dice"], trap)
for r in trap:
    if r[2] == "empty mask":
        print(f"{r[0]}: object {r[1] * 100:.1f} %, empty mask accuracy {r[3] * 100:.1f} %, IoU {r[4]:.3f}")
