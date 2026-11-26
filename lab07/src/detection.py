# Lab 7, part E: own NMS, matching with the ground truth, precision-recall curve and AP
# boxes are (x0, y0, x1, y1) in pixels, x1 and y1 excluded

import numpy as np


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(boxes, scores, threshold=0.5):
    # Step 13. sort by confidence, go down the list, drop every box with IoU > threshold with an accepted one
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) <= threshold for j in keep):
            keep.append(i)
    return keep


def match(detections, ground_truth, iou_min=0.5, sort=True):
    # Step 14. detections: list of (image, box, score); ground_truth: {image: [box, ...]}.
    # Detections are processed from the most confident one; each detection takes the ground truth box with
    # the largest IoU in the same image: TP if IoU >= 0.5 and that box is still free, otherwise FP.
    # Each ground truth box can be matched only once.
    order = np.argsort([-d[2] for d in detections], kind="stable") if sort else np.arange(len(detections))
    taken = {img: [False] * len(boxes) for img, boxes in ground_truth.items()}
    tp = np.zeros(len(detections), bool)
    best_iou = np.zeros(len(detections))
    for i in order:
        img, box, _ = detections[i]
        ious = [box_iou(box, g) for g in ground_truth.get(img, [])]
        if not ious:
            continue
        j = int(np.argmax(ious))
        best_iou[i] = ious[j]
        if ious[j] >= iou_min and not taken[img][j]:
            taken[img][j] = True
            tp[i] = True
    return tp, best_iou


def pr_curve(scores, tp, n_gt, sort=True):
    # Step 15. all detections of all images sorted by decreasing confidence, TP and FP accumulated,
    # precision and recall after every step. sort=False only to show the typical mistake.
    order = np.argsort(-np.asarray(scores), kind="stable") if sort else np.arange(len(scores))
    tp = np.asarray(tp, float)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    return ctp / n_gt, ctp / (ctp + cfp), np.asarray(scores)[order]


def average_precision(recall, precision):
    # Step 16. precision made non-increasing (max precision at any recall >= the current one),
    # then the area under the step curve: sum over the recall steps of (r_k - r_{k-1}) * p_interp(r_k)
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    p_interp = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p_interp[1:])), p_interp[1:]
