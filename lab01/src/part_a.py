# ЛР1, частина A, кроки 1-5

import csv
import os
from datetime import datetime, timedelta

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image

from utils import DATA, FIG, TMP, psnr, save_csv

np.random.seed(42)

img = cv2.imread(os.path.join(DATA, "photo.jpg"))


# Крок 1. Властивості масиву
print("Крок 1")
print("shape:", img.shape)
print("dtype:", img.dtype)
print("min:", img.min(), "max:", img.max())
print("пам'ять, МБ:", round(img.nbytes / 1024 / 1024, 2))

save_csv("a1_properties.csv", ["параметр", "значення"], [
    ["shape (H, W, C)", str(img.shape)],
    ["dtype", str(img.dtype)],
    ["min", img.min()],
    ["max", img.max()],
    ["обсяг у пам'яті, МБ", round(img.nbytes / 1024 / 1024, 2)],
])


# Крок 2. BGR і RGB
# OpenCV - BGR, matplotlib - RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(img)
ax[0].set_title("plt.imshow(img) без перетворення\n(канали R і B переставлені)")
ax[1].imshow(img_rgb)
ax[1].set_title("після cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
ax[0].axis("off")
ax[1].axis("off")
plt.savefig(os.path.join(FIG, "a1_bgr_vs_rgb.png"), dpi=150, bbox_inches="tight")
plt.close()


# Крок 3. Переповнення uint8
wrong = img + 50  # 250 + 50 = 44
right = cv2.add(img, 50)  # 250 + 50 = 255
right2 = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
print("\nКрок 3")
print("cv2.add і int16 + clip дають однаковий результат:", np.array_equal(right, right2))

overflow = (wrong < img).any(axis=2)  # хоч один канал переповнився
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_wrong = cv2.cvtColor(wrong, cv2.COLOR_BGR2GRAY)
bright_to_dark = (gray > 200) & (gray_wrong < 100)

print("пікселів з переповненням:", round(overflow.mean() * 100, 2), "%")
print("яскраві стали темними:", round(bright_to_dark.mean() * 100, 2), "%")
save_csv("a1_overflow.csv", ["метод", "пікселів з переповненням, %", "яскраві→темні, %"], [
    ["img + 50", round(overflow.mean() * 100, 2), round(bright_to_dark.mean() * 100, 2)],
    ["cv2.add(img, 50)", 0, 0],
])

# шматок неба
y1, y2, x1, x2 = 0, 384, 1388, 1772
fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
titles = ["оригінал", "img + 50 (переповнення)", "cv2.add(img, 50) (насичення)"]
for i, im in enumerate([img, wrong, right]):
    ax[i].imshow(cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    ax[i].set_title(titles[i])
    ax[i].axis("off")
plt.savefig(os.path.join(FIG, "a1_overflow.png"), dpi=150, bbox_inches="tight")
plt.close()


# Крок 4. Втрати при стисненні
print("\nКрок 4")
raw_size = img.nbytes / 1024  # КБ

cv2.imwrite(os.path.join(TMP, "photo.png"), img)
png_size = os.path.getsize(os.path.join(TMP, "photo.png")) / 1024
png_back = cv2.imread(os.path.join(TMP, "photo.png"))
print("PNG:", round(png_size), "КБ, без втрат:", np.array_equal(png_back, img))

table = [["PNG", "-", round(png_size, 1), round(raw_size / png_size, 1), "inf"]]
sizes = []
psnrs = []
qualities = list(range(5, 101, 5))  # для графіка більше точок
for q in qualities:
    path = os.path.join(TMP, f"photo_q{q}.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    size = os.path.getsize(path) / 1024
    back = cv2.imread(path)
    p = psnr(img, back)
    sizes.append(size)
    psnrs.append(p)
    if q in [95, 75, 50, 25, 10]:
        print(f"JPEG q={q}: {size:.1f} КБ, PSNR = {p:.2f} дБ (cv2.PSNR = {cv2.PSNR(img, back):.2f})")
        table.append(["JPEG", q, round(size, 1), round(raw_size / size, 1), round(p, 2)])

save_csv("a2_compression.csv", ["формат", "якість", "розмір, КБ", "стиснення, разів", "PSNR, дБ"], table)

plt.figure(figsize=(7, 4.5))
plt.plot(sizes, psnrs, "o-", label="JPEG, якість 5...100")
for q, s, p in zip(qualities, sizes, psnrs):
    if q in [95, 75, 50, 25, 10]:
        plt.annotate(f"q={q}", (s, p), xytext=(5, -12), textcoords="offset points", fontsize=8)
plt.axvline(png_size, color="gray", linestyle="--", label=f"PNG ({png_size:.0f} КБ, без втрат)")
plt.xscale("log")
plt.xlabel("розмір файлу, КБ (логарифмічна шкала)")
plt.ylabel("PSNR, дБ")
plt.title("Залежність PSNR від розміру файлу JPEG")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FIG, "a2_psnr_vs_size.png"), dpi=150, bbox_inches="tight")
plt.close()

# артефакти
y1, y2, x1, x2 = 600, 856, 900, 1156
fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
for i, q in enumerate([None, 50, 10]):
    if q is None:
        part = img[y1:y2, x1:x2]
        ax[i].set_title("оригінал")
    else:
        part = cv2.imread(os.path.join(TMP, f"photo_q{q}.jpg"))[y1:y2, x1:x2]
        ax[i].set_title(f"JPEG q={q}")
    ax[i].imshow(cv2.cvtColor(part, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax[i].axis("off")
plt.savefig(os.path.join(FIG, "a2_jpeg_artifacts.png"), dpi=150, bbox_inches="tight")
plt.close()


# Крок 5. EXIF
print("\nКрок 5")


def read_exif(path):
    image = Image.open(path)
    exif = image.getexif()
    tags = {}
    for key, value in exif.items():
        tags[ExifTags.TAGS.get(key, key)] = value
    # витримка, ISO - в розділі Exif
    for key, value in exif.get_ifd(ExifTags.IFD.Exif).items():
        tags[ExifTags.TAGS.get(key, key)] = value
    return tags


rows = []
for name in ["photo.jpg", "texture.jpg", "object_light1.jpg", "object_light2.jpg"]:
    t = read_exif(os.path.join(DATA, name))
    exposure = f"1/{round(1 / float(t['ExposureTime']))} с"
    row = [name, t["Model"], f"{float(t['FocalLength'])} мм", f"{t['FocalLengthIn35mmFilm']} мм",
           f"f/{float(t['FNumber'])}", exposure, t["ISOSpeedRatings"], t["DateTimeOriginal"]]
    print(row)
    rows.append(row)

save_csv("a3_exif.csv", ["файл", "камера", "фокусна відстань", "екв. 35 мм", "діафрагма",
                         "витримка", "ISO", "дата"], rows)

# кадри набору (лекція 1: EXIF кожного кадру - до будь-якої обробки)
# у кадрів з відео EXIF немає - беру метадані відео, які зберіг extract_frames.py
video_info = {}
for name in os.listdir(os.path.join(DATA, "dataset")):
    if name.endswith("_frames.csv"):
        with open(os.path.join(DATA, "dataset", name), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_info[row["файл"]] = row

rows = []
for name in sorted(os.listdir(os.path.join(DATA, "dataset"))):
    if not name.lower().endswith((".jpg", ".jpeg")):
        continue
    t = read_exif(os.path.join(DATA, "dataset", name))
    if "ExposureTime" in t:
        rows.append([name, t["Model"], f"{t['FocalLengthIn35mmFilm']} мм", f"1/{round(1 / float(t['ExposureTime']))} с",
                     t["ISOSpeedRatings"], t["DateTimeOriginal"], "фото"])
    else:
        v = video_info[name]
        # час кадру = початок запису + час у відео
        moment = datetime.fromisoformat(v["початок запису"]) + timedelta(seconds=float(v["час у відео, с"]))
        rows.append([name, v["камера"], "-", "-", "-", moment.strftime("%Y:%m:%d %H:%M:%S.%f")[:-4],
                     f"кадр {v['кадр']} з {v['відео']} (у відео немає витримки й ISO)"])
print("кадрів набору:", len(rows))
save_csv("a3_exif_dataset.csv", ["файл", "камера", "екв. 35 мм", "витримка", "ISO", "дата", "джерело"], rows)
