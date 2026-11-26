# Lab 7, part D: agreement between two independent annotations of the same image

import os

import cv2
import numpy as np

from utils import ANNOTATIONS, FIG, dice, fit, hstack, iou, label, load_image, load_mask, polygons_to_mask, \
    save_csv

# second annotations: data/annotations/second/<name>.json (same format as the first ones)
SECOND = os.path.join(ANNOTATIONS, "second")

rows = []
tiles = []
for file in sorted(os.listdir(SECOND)):
    name = file[:-5]
    image = load_image(name)
    first = load_mask(name)
    second = polygons_to_mask(os.path.join(SECOND, file), image.shape) > 0
    disagree = first ^ second
    rows.append([name, int(first.sum()), int(second.sum()), int(np.logical_and(first, second).sum()),
                 int(disagree.sum()), round(iou(first, second), 4), round(dice(first, second), 4)])
    print(f"{name}: IoU between the two annotations {iou(first, second):.3f}, Dice {dice(first, second):.3f}, "
          f"{disagree.sum()} px disagree ({first.sum()} and {second.sum()} px)")

    # picture: crop around the object, green = only the first, red = only the second, yellow = both
    ys, xs = np.nonzero(first | second)
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, image.shape[0])
    x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, image.shape[1])
    vis = image.copy()
    for m, color in ((first & ~second, (0, 200, 0)), (second & ~first, (0, 0, 255)), (first & second, (0, 220, 255))):
        vis[m] = (vis[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    tiles.append(label(fit(vis[y0:y1, x0:x1], 360), f"{name}: IoU {iou(first, second):.3f}", 0.6))

save_csv("agreement.csv", ["image", "first, px", "second, px", "both, px", "disagree, px", "IoU", "Dice"], rows)
cv2.imwrite(os.path.join(FIG, "agreement.jpg"), hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
