# object mask with GrabCut
# python make_mask.py object_light1.jpg object_mask.png 360 380 800 1330   (x y w h)

import os
import sys

import cv2
import numpy as np

from utils import DATA, TMP

image_name, mask_name = sys.argv[1], sys.argv[2]
x, y, w, h = [int(v) for v in sys.argv[3:7]]

img = cv2.imread(os.path.join(DATA, image_name))
mask = np.zeros(img.shape[:2], dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)
cv2.grabCut(img, mask, (x, y, w, h), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 0, 2 - background; 1, 3 - object
result = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
cv2.imwrite(os.path.join(DATA, mask_name), result)
print("object area:", round((result > 0).mean() * 100, 1), "% of the frame")

# preview to check the mask
preview = img.copy()
preview[result == 0] = preview[result == 0] // 3
cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
cv2.imwrite(os.path.join(TMP, mask_name.replace(".png", "_preview.jpg")), preview)
