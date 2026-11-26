# Lab 7, part E data: Penn-Fudan pedestrian database (170 images with boxes around the pedestrians)

import os
import urllib.request
import zipfile

from utils import DATA, PENNFUDAN

URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
ZIP = os.path.join(DATA, "PennFudanPed.zip")

if not os.path.exists(PENNFUDAN):
    if not os.path.exists(ZIP):
        print("downloading", URL, "(54 MB)")
        urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA)
    os.remove(ZIP)

images = sorted(os.listdir(os.path.join(PENNFUDAN, "PNGImages")))
print(f"{len(images)} images in {PENNFUDAN}")
