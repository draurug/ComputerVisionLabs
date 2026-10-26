# Lab 1, part B, steps 8-11
# bounds = percentiles 1..99 of the object pixels on frame 1, the same bounds are used on frame 2

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DATA, FIG, accuracy, iou, save_csv

np.random.seed(42)

frame1 = cv2.imread(os.path.join(DATA, "object_light1.jpg"))   # window light
frame2 = cv2.imread(os.path.join(DATA, "object_light2.jpg"))   # lamp
mask1 = cv2.imread(os.path.join(DATA, "object_mask.png"), cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread(os.path.join(DATA, "object_mask2.png"), cv2.IMREAD_GRAYSCALE)  # phone moved a bit
print("object area:", round((mask1 > 0).mean() * 100, 1), "% of the frame")


def find_bounds(image, mask, channels, q=1):
    # percentiles q..100-q; other channels stay 0..255
    low = np.array([0, 0, 0], dtype=np.uint8)
    high = np.array([255, 255, 255], dtype=np.uint8)
    for ch in channels:
        values = image[:, :, ch][mask > 0]
        low[ch] = int(np.percentile(values, q))
        high[ch] = int(np.percentile(values, 100 - q))
    return low, high


spaces = [
    ["RGB", None, [0, 1, 2]],
    ["HSV", cv2.COLOR_BGR2HSV, [0, 1, 2]],
    ["HSV (H, S only)", cv2.COLOR_BGR2HSV, [0, 1]],
    ["Lab (a, b only)", cv2.COLOR_BGR2Lab, [1, 2]],
]
# H in OpenCV is 0..179, a red object would need two intervals

bounds_rows = []
metric_rows = []
results = {}

for name, code, channels in spaces:
    if code is None:
        conv1, conv2 = frame1, frame2
    else:
        conv1 = cv2.cvtColor(frame1, code)
        conv2 = cv2.cvtColor(frame2, code)

    # bounds from frame 1
    low, high = find_bounds(conv1, mask1, channels)
    bounds_rows.append([name, str(low.tolist()), str(high.tolist())])

    pred1 = cv2.inRange(conv1, low, high)
    pred2 = cv2.inRange(conv2, low, high)
    results[name] = (pred1, pred2)

    iou1, iou2 = iou(pred1, mask1), iou(pred2, mask2)
    acc1, acc2 = accuracy(pred1, mask1), accuracy(pred2, mask2)
    metric_rows.append([name, round(iou1, 3), round(acc1, 3), round(iou2, 3), round(acc2, 3)])
    print(f"{name:16s} bounds {low.tolist()}..{high.tolist()}  frame 1: IoU = {iou1:.3f}  frame 2: IoU = {iou2:.3f}")

# Step 11. Table
save_csv("b_thresholds.csv", ["space", "lower bounds", "upper bounds"], bounds_rows)
save_csv("b_metrics.csv", ["space", "IoU, frame 1", "accuracy, frame 1", "IoU, frame 2", "accuracy, frame 2"],
         metric_rows)

# accuracy of an empty mask, to compare with
print("accuracy of an empty mask on frame 2:", round(accuracy(np.zeros_like(mask2), mask2), 3))

# does the conclusion depend on the percentiles
print("\nIoU on frame 2 for different percentiles")
sens_rows = []
for name, code, channels in spaces:
    if code is None:
        conv1, conv2 = frame1, frame2
    else:
        conv1 = cv2.cvtColor(frame1, code)
        conv2 = cv2.cvtColor(frame2, code)
    row = [name]
    for q in [1, 2, 5]:
        low, high = find_bounds(conv1, mask1, channels, q)
        row.append(float(round(iou(cv2.inRange(conv2, low, high), mask2), 3)))
    print(name, row[1:])
    sens_rows.append(row)
save_csv("b_sensitivity.csv", ["space", "1-99", "2-98", "5-95"], sens_rows)


# figures
fig, ax = plt.subplots(1, 4, figsize=(16, 5))
pictures = [cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), mask1, mask2]
titles = ["frame 1 (daylight)", "frame 2 (lamp)", "mask of frame 1", "mask of frame 2"]
for i in range(4):
    ax[i].imshow(pictures[i], cmap="gray")
    ax[i].set_title(titles[i])
    ax[i].axis("off")
plt.savefig(os.path.join(FIG, "b_inputs.png"), dpi=150, bbox_inches="tight")
plt.close()


def errors_picture(pred, gt):
    # green - correct, red - false positive, blue - missed
    pred = pred > 0
    gt = gt > 0
    pic = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pic[pred & gt] = [0, 170, 0]
    pic[pred & ~gt] = [230, 30, 30]
    pic[~pred & gt] = [40, 80, 255]
    return pic


fig, ax = plt.subplots(2, len(spaces), figsize=(3.2 * len(spaces), 8))
for i, (name, code, channels) in enumerate(spaces):
    pred1, pred2 = results[name]
    ax[0, i].imshow(errors_picture(pred1, mask1))
    ax[0, i].set_title(f"{name}, frame 1\nIoU = {iou(pred1, mask1):.2f}", fontsize=10)
    ax[1, i].imshow(errors_picture(pred2, mask2))
    ax[1, i].set_title(f"{name}, frame 2\nIoU = {iou(pred2, mask2):.2f}", fontsize=10)
    ax[0, i].axis("off")
    ax[1, i].axis("off")
fig.suptitle("green - correct, red - false positive, blue - missed")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "b_masks.png"), dpi=150, bbox_inches="tight")
plt.close()

names = [row[0] for row in metric_rows]
x = np.arange(len(names))
plt.figure(figsize=(9, 4.5))
plt.bar(x - 0.2, [row[1] for row in metric_rows], 0.4, label="frame 1 (bounds taken here)")
plt.bar(x + 0.2, [row[3] for row in metric_rows], 0.4, label="frame 2 (same bounds)")
plt.xticks(x, names)
plt.ylabel("IoU with the reference mask")
plt.ylim(0, 1)
plt.title("Threshold stability under a lighting change")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FIG, "b_iou.png"), dpi=150, bbox_inches="tight")
plt.close()
