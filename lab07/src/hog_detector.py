# Lab 7, part E: HOG + linear SVM people detector on 30 Penn-Fudan images, own NMS, matching,
# precision-recall curve and AP

import os
import re
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog as sk_hog

from detection import average_precision, box_iou, match, nms, pr_curve
from utils import FIG, PENNFUDAN, fit, hstack, label, save_csv, vstack

np.random.seed(42)
# one thread: the order of the raw windows (and of windows with equal scores) is then the same in every run
cv2.setNumThreads(1)

# Step 12. 30 of the 170 images, evenly spread over the sorted list (both the Fudan and the Penn part)
ALL = sorted(os.listdir(os.path.join(PENNFUDAN, "PNGImages")))
SELECT = [ALL[i] for i in np.linspace(0, len(ALL) - 1, 30).round().astype(int)]

# detectMultiScale parameters. hitThreshold below 0 keeps weak windows too: they go to the end of the
# ranking and only extend the curve to higher recall, the beginning of the curve does not change
HIT = -0.5
WIN_STRIDE = (8, 8)
PADDING = (8, 8)
SCALE = 1.05
NMS_IOU = 0.5


def read_boxes(name):
    # "(Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)", 1-based and inclusive -> 0-based, x1/y1 excluded
    path = os.path.join(PENNFUDAN, "Annotation", name.replace(".png", ".txt"))
    with open(path, encoding="latin-1") as f:
        text = f.read()
    return [(int(a) - 1, int(b) - 1, int(c), int(d))
            for a, b, c, d in re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", text)]


def shrink(box):
    # the 64x128 window has a margin around the person; the OpenCV sample peopledetect.py draws the box
    # 15 % narrower on each side and 5 % lower on each side
    x0, y0, x1, y1 = box
    pw, ph = 0.15 * (x1 - x0), 0.05 * (y1 - y0)
    return (x0 + pw, y0 + ph, x1 - pw, y1 - ph)


hog = cv2.HOGDescriptor()                                       # 64x128 window, 8x8 cells, 16x16 blocks, 9 bins
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

images, gt, raw, grouped = {}, {}, {}, {}
times = []
for name in SELECT:
    image = cv2.imread(os.path.join(PENNFUDAN, "PNGImages", name))
    images[name] = image
    gt[name] = read_boxes(name)
    t = time.perf_counter()
    rects, weights = hog.detectMultiScale(image, hitThreshold=HIT, winStride=WIN_STRIDE, padding=PADDING,
                                          scale=SCALE, groupThreshold=0)     # 0 = no grouping, raw windows
    times.append(time.perf_counter() - t)
    raw[name] = [((x, y, x + w, y + h), float(s)) for (x, y, w, h), s in zip(rects, np.ravel(weights))]
    # for reference: the built-in grouping of OpenCV (groupRectangles, default groupThreshold = 2)
    rects, weights = hog.detectMultiScale(image, hitThreshold=HIT, winStride=WIN_STRIDE, padding=PADDING,
                                          scale=SCALE, groupThreshold=2)
    grouped[name] = [((x, y, x + w, y + h), float(s)) for (x, y, w, h), s in zip(rects, np.ravel(weights))]

n_gt = sum(len(b) for b in gt.values())
n_raw = sum(len(v) for v in raw.values())
print(f"{len(SELECT)} images, {n_gt} annotated pedestrians, {n_raw} raw windows, "
      f"{np.mean(times) * 1000:.0f} ms per image (median size "
      f"{int(np.median([i.shape[1] for i in images.values()]))}x{int(np.median([i.shape[0] for i in images.values()]))})")


def after_nms(threshold, box_fn=lambda b: b, source=raw):
    dets = []
    for name in SELECT:
        # the boxes are corrected first, NMS works on the boxes that are evaluated
        boxes = [box_fn(b) for b, _ in source[name]]
        scores = [s for _, s in source[name]]
        keep = range(len(boxes)) if threshold is None else nms(boxes, scores, threshold)
        dets += [(name, boxes[i], scores[i]) for i in keep]
    return dets


def evaluate(dets, sort=True):
    tp, best = match(dets, gt, 0.5, sort=sort)
    recall, precision, scores = pr_curve([d[2] for d in dets], tp, n_gt, sort=sort)
    ap, p_interp = average_precision(recall, precision)
    return dict(dets=dets, tp=tp, best=best, recall=recall, precision=precision, scores=scores, ap=ap,
                p_interp=p_interp)


variants = []
for box_name, fn in (("as returned", lambda b: b), ("shrunk", shrink)):
    variants += [("no NMS", box_name, None, fn, raw), ("NMS 0.3", box_name, 0.3, fn, raw),
                 ("NMS 0.5", box_name, 0.5, fn, raw), ("NMS 0.7", box_name, 0.7, fn, raw),
                 ("OpenCV grouping", box_name, None, fn, grouped)]
res = {}
rows = []
for nms_name, box_name, thr, fn, source in variants:
    r = evaluate(after_nms(thr, fn, source))
    res[(nms_name, box_name)] = r
    f1 = 2 * r["precision"] * r["recall"] / np.maximum(r["precision"] + r["recall"], 1e-12)
    k = int(np.argmax(f1))
    rows.append([nms_name, box_name, len(r["dets"]), int(r["tp"].sum()), round(r["recall"][-1], 4),
                 round(r["ap"], 4), round(f1[k], 4), round(r["precision"][k], 4), round(r["recall"][k], 4),
                 round(r["scores"][k], 3)])
    print(f"{nms_name:16s} {box_name:12s} {len(r['dets']):5d} detections, TP {int(r['tp'].sum()):3d}, "
          f"max recall {r['recall'][-1]:.3f}, AP {r['ap']:.3f}; best F1 {f1[k]:.3f} "
          f"(P {r['precision'][k]:.3f}, R {r['recall'][k]:.3f}, score >= {r['scores'][k]:.2f})")
save_csv("detection_summary.csv", ["suppression", "boxes", "detections", "TP", "max recall", "AP", "best F1",
                                   "precision at best F1", "recall at best F1", "score at best F1"], rows)

# main variant: NMS 0.5 on the boxes shrunk as in the OpenCV sample; the boxes as returned for comparison
main = res[("NMS 0.5", "shrunk")]
asr = res[("NMS 0.5", "as returned")]

# the curve of the main variant, point by point
save_csv("pr_curve.csv", ["rank", "score", "recall", "precision", "precision, non-increasing"],
         [[i + 1, round(s, 4), round(r, 4), round(p, 4), round(q, 4)] for i, (s, r, p, q) in
          enumerate(zip(main["scores"], main["recall"], main["precision"], main["p_interp"]))])

# all detections of the main variant with their match
order = np.argsort([-d[2] for d in main["dets"]], kind="stable")
save_csv("detections.csv", ["image", "x0", "y0", "x1", "y1", "score", "best IoU with ground truth", "TP"],
         [[main["dets"][i][0], *[int(round(v)) for v in main["dets"][i][1]], round(main["dets"][i][2], 4),
           round(main["best"][i], 4), int(main["tp"][i])] for i in order])

# typical mistake: the curve without sorting by confidence. In the order the detections were produced
# (image by image) and in 200 random orders
unsorted = evaluate(main["dets"], sort=False)
rng = np.random.default_rng(42)
shuffled = []
for _ in range(200):
    perm = rng.permutation(len(main["dets"]))
    shuffled.append(evaluate([main["dets"][i] for i in perm], sort=False)["ap"])
shuffled = np.array(shuffled)
print(f"AP sorted {main['ap']:.3f}, unsorted (image order) {unsorted['ap']:.3f}, "
      f"random order {shuffled.mean():.3f} +- {shuffled.std():.3f} [{shuffled.min():.3f}, {shuffled.max():.3f}]")

# why the boxes matter: IoU of every true pedestrian with its best detection
best_raw = [max([box_iou(d[1], g) for d in asr["dets"] if d[0] == n] or [0]) for n in SELECT for g in gt[n]]
best_shr = [max([box_iou(d[1], g) for d in main["dets"] if d[0] == n] or [0]) for n in SELECT for g in gt[n]]
gt_ratio = np.median([(g[2] - g[0]) / (g[3] - g[1]) for n in SELECT for g in gt[n]])
print(f"best IoU per pedestrian: as returned median {np.median(best_raw):.3f}, {np.mean(np.array(best_raw) >= 0.5) * 100:.0f} % "
      f">= 0.5; shrunk median {np.median(best_shr):.3f}, {np.mean(np.array(best_shr) >= 0.5) * 100:.0f} % >= 0.5; "
      f"width/height of the annotated boxes median {gt_ratio:.2f} (window 0.50)")

save_csv("ap_extra.csv", ["quantity", "value"], [
    ["images", len(SELECT)], ["annotated pedestrians", n_gt], ["raw windows", n_raw],
    ["ms per image", round(np.mean(times) * 1000, 1)],
    ["AP, sorted (NMS 0.5, shrunk boxes)", round(main["ap"], 4)],
    ["AP, not sorted (image order)", round(unsorted["ap"], 4)],
    ["AP, random order, mean", round(shuffled.mean(), 4)], ["AP, random order, std", round(shuffled.std(), 4)],
    ["AP, random order, min", round(shuffled.min(), 4)], ["AP, random order, max", round(shuffled.max(), 4)],
    ["best IoU per pedestrian, boxes as returned, median", round(float(np.median(best_raw)), 4)],
    ["share of pedestrians with best IoU >= 0.5, as returned", round(float(np.mean(np.array(best_raw) >= 0.5)), 4)],
    ["best IoU per pedestrian, shrunk, median", round(float(np.median(best_shr)), 4)],
    ["share of pedestrians with best IoU >= 0.5, shrunk", round(float(np.mean(np.array(best_shr) >= 0.5)), 4)],
    ["annotated box width / height, median", round(float(gt_ratio), 3)],
])

# ---- figures ----

# PR curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for key, color in ((("NMS 0.5", "as returned"), "C0"), (("NMS 0.5", "shrunk"), "C1")):
    r = res[key]
    ax[0].plot(r["recall"], r["precision"], color=color, alpha=0.35, lw=1)
    ax[0].step(r["recall"], r["p_interp"], color=color, where="post", lw=2,
               label=f"boxes {key[1]}: AP = {r['ap']:.3f}")
ax[0].plot(unsorted["recall"], unsorted["precision"], color="gray", lw=1, ls=":",
           label=f"not sorted by score (mistake): AP = {unsorted['ap']:.3f}")
ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")
ax[0].set_title("HOG + SVM, NMS 0.5, IoU >= 0.5\n(thin = raw precision, thick = non-increasing)")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1.02)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="lower left")
for (nms_name, box_name), r in res.items():
    if box_name == "shrunk" and nms_name.startswith(("NMS", "no")):
        ax[1].step(r["recall"], r["p_interp"], where="post", lw=1.6,
                   label=f"{nms_name}: AP = {r['ap']:.3f}, {len(r['dets'])} det.")
ax[1].set_xlabel("recall")
ax[1].set_ylabel("precision (non-increasing)")
ax[1].set_title("Effect of the NMS threshold (shrunk boxes)")
ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, 1.02)
ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "pr_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# HOG descriptor of one annotated pedestrian
name = SELECT[0]
x0, y0, x1, y1 = gt[name][0]
h = y1 - y0
cx = (x0 + x1) / 2
# window with the same 1:2 shape and margin as the 64x128 training window (person ~ 3/4 of the height)
H = h / 0.75
W = H / 2
X0, Y0 = int(max(cx - W / 2, 0)), int(max(y0 - (H - h) / 2, 0))
crop = images[name][Y0:int(Y0 + H), X0:int(X0 + W)]
crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
desc = hog.compute(crop)
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, vis = sk_hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
print(f"HOG descriptor of a 64x128 window: {desc.size} values (7x15 blocks x 4 cells x 9 bins)")
fig, ax = plt.subplots(1, 3, figsize=(7.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax[0].set_title("window 64×128")
ax[1].imshow(vis, cmap="gray")
ax[1].set_title("HOG, cells 8×8")
for a in ax[:2]:
    a.set_xticks([0, 32, 63])
    a.set_yticks([0, 64, 127])
    a.set_xlabel("x, px")
ax[0].set_ylabel("y, px")
ax[2].plot(np.ravel(desc), lw=0.5)
ax[2].set_xlabel("component of the descriptor")
ax[2].set_ylabel("value")
ax[2].set_title(f"descriptor, {desc.size} values")
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "hog.png"), dpi=150, bbox_inches="tight")
plt.close()


def draw(name, dets, tp=None, threshold=None):
    out = images[name].copy()
    for g in gt[name]:
        cv2.rectangle(out, g[:2], g[2:], (0, 200, 0), 3)
    for i, (n, b, s) in enumerate(dets):
        if n != name or (threshold is not None and s < threshold):
            continue
        color = (255, 120, 0) if tp is None else ((255, 120, 0) if tp[i] else (0, 0, 255))
        b = [int(round(v)) for v in b]
        cv2.rectangle(out, b[:2], b[2:], color, 1 if tp is None else 2)
        if tp is not None:
            cv2.putText(out, f"{s:.2f}", (b[0] + 3, b[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


# NMS effect on one image: raw windows and after NMS 0.5
name = SELECT[3]
raw_d = [(name, b, s) for b, s in raw[name]]
nms_d = [d for d in main["dets"] if d[0] == name]
cv2.imwrite(os.path.join(FIG, "nms_effect.jpg"),
            hstack([label(fit(draw(name, raw_d), 380), f"raw: {len(raw_d)}", 0.6),
                    label(fit(draw(name, nms_d), 380), f"NMS 0.5: {len(nms_d)}", 0.6)]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])

# examples at the score of the best F1 (shrunk boxes): green = ground truth, blue = TP, red = FP
k = int(np.argmax(2 * main["precision"] * main["recall"] / np.maximum(main["precision"] + main["recall"], 1e-12)))
thr = main["scores"][k]
tiles = [label(fit(draw(n, main["dets"], main["tp"], thr), 300), n[:-4], 0.55) for n in SELECT[::5][:6]]
cv2.imwrite(os.path.join(FIG, "detections.jpg"), vstack([hstack(tiles[:3]), hstack(tiles[3:])]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"examples drawn at score >= {thr:.2f}")
