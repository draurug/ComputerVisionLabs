# ЛР1, частина B, кроки 8-11
# межі - перцентилі 1..99 пікселів об'єкта на кадрі 1, ті самі межі на кадрі 2

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DATA, FIG, accuracy, iou, save_csv

np.random.seed(42)

frame1 = cv2.imread(os.path.join(DATA, "object_light1.jpg"))   # вікно
frame2 = cv2.imread(os.path.join(DATA, "object_light2.jpg"))   # лампа
mask1 = cv2.imread(os.path.join(DATA, "object_mask.png"), cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread(os.path.join(DATA, "object_mask2.png"), cv2.IMREAD_GRAYSCALE)  # телефон зсунувся
print("об'єкт займає", round((mask1 > 0).mean() * 100, 1), "% кадру")


def find_bounds(image, mask, channels, q=1):
    # перцентилі q..100-q; інші канали 0..255
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
    ["HSV (лише H, S)", cv2.COLOR_BGR2HSV, [0, 1]],
    ["Lab (лише a, b)", cv2.COLOR_BGR2Lab, [1, 2]],
]
# H в OpenCV 0..179, для червоного треба було б два інтервали

bounds_rows = []
metric_rows = []
results = {}

for name, code, channels in spaces:
    if code is None:
        conv1, conv2 = frame1, frame2
    else:
        conv1 = cv2.cvtColor(frame1, code)
        conv2 = cv2.cvtColor(frame2, code)

    # межі по кадру 1
    low, high = find_bounds(conv1, mask1, channels)
    bounds_rows.append([name, str(low.tolist()), str(high.tolist())])

    pred1 = cv2.inRange(conv1, low, high)
    pred2 = cv2.inRange(conv2, low, high)
    results[name] = (pred1, pred2)

    iou1, iou2 = iou(pred1, mask1), iou(pred2, mask2)
    acc1, acc2 = accuracy(pred1, mask1), accuracy(pred2, mask2)
    metric_rows.append([name, round(iou1, 3), round(acc1, 3), round(iou2, 3), round(acc2, 3)])
    print(f"{name:16s} межі {low.tolist()}..{high.tolist()}  кадр 1: IoU = {iou1:.3f}  кадр 2: IoU = {iou2:.3f}")

# Крок 11. Таблиця
save_csv("b_thresholds.csv", ["простір", "нижні межі", "верхні межі"], bounds_rows)
save_csv("b_metrics.csv", ["простір", "IoU, кадр 1", "accuracy, кадр 1", "IoU, кадр 2", "accuracy, кадр 2"],
         metric_rows)

# accuracy порожньої маски
print("accuracy порожньої маски на кадрі 2:", round(accuracy(np.zeros_like(mask2), mask2), 3))

# чи залежить висновок від вибору перцентилів
print("\nIoU на кадрі 2 при різних перцентилях")
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
save_csv("b_sensitivity.csv", ["простір", "1-99", "2-98", "5-95"], sens_rows)


# рисунки
fig, ax = plt.subplots(1, 4, figsize=(16, 5))
pictures = [cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), mask1, mask2]
titles = ["кадр 1 (денне світло)", "кадр 2 (лампа)", "маска кадру 1", "маска кадру 2"]
for i in range(4):
    ax[i].imshow(pictures[i], cmap="gray")
    ax[i].set_title(titles[i])
    ax[i].axis("off")
plt.savefig(os.path.join(FIG, "b_inputs.png"), dpi=150, bbox_inches="tight")
plt.close()


def errors_picture(pred, gt):
    # зелений - ок, червоний - зайве, синій - пропущено
    pred = pred > 0
    gt = gt > 0
    pic = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pic[pred & gt] = [0, 170, 0]
    pic[pred & ~gt] = [230, 30, 30]
    pic[~pred & gt] = [40, 80, 255]
    return pic


fig, ax = plt.subplots(len(spaces), 2, figsize=(8, 3.2 * len(spaces)))
for i, (name, code, channels) in enumerate(spaces):
    pred1, pred2 = results[name]
    ax[i, 0].imshow(errors_picture(pred1, mask1))
    ax[i, 0].set_title(f"{name}, кадр 1: IoU = {iou(pred1, mask1):.2f}", fontsize=9)
    ax[i, 1].imshow(errors_picture(pred2, mask2))
    ax[i, 1].set_title(f"{name}, кадр 2: IoU = {iou(pred2, mask2):.2f}", fontsize=9)
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
fig.suptitle("зелений - правильно, червоний - зайве, синій - пропущено")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "b_masks.png"), dpi=150, bbox_inches="tight")
plt.close()

names = [row[0] for row in metric_rows]
x = np.arange(len(names))
plt.figure(figsize=(9, 4.5))
plt.bar(x - 0.2, [row[1] for row in metric_rows], 0.4, label="кадр 1 (тут підбирались межі)")
plt.bar(x + 0.2, [row[3] for row in metric_rows], 0.4, label="кадр 2 (ті самі межі)")
plt.xticks(x, names)
plt.ylabel("IoU з еталонною маскою")
plt.ylim(0, 1)
plt.title("Стабільність порогів при зміні освітлення")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FIG, "b_iou.png"), dpi=150, bbox_inches="tight")
plt.close()
