# спільне для всіх частин

import csv
import os

import numpy as np

# папка lab01
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")
TMP = os.path.join(RESULTS, "tmp")  # тимчасові jpg/png

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def psnr(original, test):
    # 10 * log10(255^2 / MSE)
    mse = np.mean((original.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


def iou(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union


def accuracy(pred, gt):
    return np.mean((pred > 0) == (gt > 0))


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
