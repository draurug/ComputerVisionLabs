# маска об'єкта через GrabCut
# python make_mask.py object_light1.jpg object_mask.png 360 380 800 1330   (x y w h)

import os
import sys

import cv2
import numpy as np

from utils import DATA, FIG

image_name, mask_name = sys.argv[1], sys.argv[2]
x, y, w, h = [int(v) for v in sys.argv[3:7]]

img = cv2.imread(os.path.join(DATA, image_name))
mask = np.zeros(img.shape[:2], dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)
cv2.grabCut(img, mask, (x, y, w, h), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 0, 2 - фон; 1, 3 - об'єкт
result = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
cv2.imwrite(os.path.join(DATA, mask_name), result)
print("об'єкт займає", round((result > 0).mean() * 100, 1), "% кадру")

# превью
preview = img.copy()
preview[result == 0] = preview[result == 0] // 3
cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
cv2.imwrite(os.path.join(FIG, mask_name.replace(".png", "_preview.jpg")), preview)
