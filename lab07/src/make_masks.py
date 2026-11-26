# Lab 7, part A: polygons from data/annotations -> binary PNG masks, step 2 check

import os

import cv2
import numpy as np

from utils import ANNOTATIONS, FIG, MASKS, NAMES, fit, hstack, label, load_image, overlay, polygons_to_mask, \
    save_csv, vstack

rows = []
tiles = []
for name in NAMES:
    image = load_image(name)
    mask = polygons_to_mask(os.path.join(ANNOTATIONS, name + ".json"), image.shape)
    cv2.imwrite(os.path.join(MASKS, name + ".png"), mask)

    # Step 2 check: same size as the image and exactly two values (0 = background, 255 = object)
    saved = cv2.imread(os.path.join(MASKS, name + ".png"), cv2.IMREAD_GRAYSCALE)
    values = np.unique(saved)
    assert saved.shape == image.shape[:2], name
    assert list(values) == [0, 255], (name, values)
    share = (saved == 255).mean()
    h, w = saved.shape
    rows.append([name, f"{w}x{h}", len(values), round(share, 4)])
    print(f"{name:13s} {w}x{h}, values {values.tolist()}, object {share * 100:.1f} % of the image")
    tiles.append(label(fit(overlay(image, saved > 0, (0, 0, 255)), 300), name, 0.6))

save_csv("masks.csv", ["image", "size", "mask values", "object share"], rows)
cv2.imwrite(os.path.join(FIG, "annotation.jpg"), vstack([hstack(tiles[:5]), hstack(tiles[5:])]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
