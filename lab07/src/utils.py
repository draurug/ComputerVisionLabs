# common stuff: paths, the 10 own images, masks, metrics, csv

import csv
import json
import os

import cv2
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
IMAGES = os.path.join(DATA, "images")
ANNOTATIONS = os.path.join(DATA, "annotations")
MASKS = os.path.join(DATA, "masks")
PENNFUDAN = os.path.join(DATA, "PennFudanPed")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")

for d in (MASKS, RESULTS, FIG):
    os.makedirs(d, exist_ok=True)

# Step 1. 10 own images with one object each (iPhone 14 Pro, downscaled to 1024 px on the long side)
NAMES = ["honey_window", "honey_lamp", "hoop", "sheet_01", "sheet_05", "sheet_09",
         "house_0663", "house_0664", "house_2217", "house_0252"]


def load_image(name):
    return cv2.imread(os.path.join(IMAGES, name + ".jpg"))


def load_mask(name):
    # object = True
    return cv2.imread(os.path.join(MASKS, name + ".png"), cv2.IMREAD_GRAYSCALE) > 127


def polygons_to_mask(path, shape):
    # annotation = list of polygons in pixel coordinates (labelme-like json), filled with 255
    with open(path, encoding="utf-8") as f:
        polygons = json.load(f)["polygons"]
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(p, np.int32) for p in polygons], 255)
    return mask


# Step 7. metrics, own implementation on boolean masks

def iou(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0      # both empty
    return np.logical_and(a, b).sum() / union


def dice(a, b):
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2 * np.logical_and(a, b).sum() / total


def pixel_accuracy(pred, gt):
    return (pred == gt).mean()


def border_fraction(mask):
    # share of the image border pixels that are in the mask
    border = np.concatenate([mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]])
    return border.mean()


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def overlay(image, mask, color=(0, 0, 255), alpha=0.45):
    # mask drawn over the image with its contour
    out = image.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def label(image, text, scale=0.8):
    out = image.copy()
    cv2.putText(out, text, (8, int(30 * scale + 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(out, text, (8, int(30 * scale + 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return out


def fit(image, height):
    s = height / image.shape[0]
    return cv2.resize(image, (round(image.shape[1] * s), height), interpolation=cv2.INTER_AREA)


def hstack(images, gap=4):
    # side by side with a white gap between the images
    h = max(im.shape[0] for im in images)
    padded = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0 if i == len(images) - 1 else gap,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for i, im in enumerate(images)]
    return np.hstack(padded)


def vstack(images, gap=4):
    w = max(im.shape[1] for im in images)
    padded = [cv2.copyMakeBorder(im, 0, 0 if i == len(images) - 1 else gap, 0, w - im.shape[1],
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for i, im in enumerate(images)]
    return np.vstack(padded)
