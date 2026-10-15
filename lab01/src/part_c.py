# ЛР1, частина C, кроки 12-14

import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

from utils import DATA, FIG, psnr, save_csv

np.random.seed(42)

photo = cv2.imread(os.path.join(DATA, "photo.jpg"))
texture = cv2.imread(os.path.join(DATA, "texture.jpg"))


# Крок 12. Аліасинг
print("Крок 12")


def downscale_4_ways(image):
    w, h = image.shape[1] // 4, image.shape[0] // 4
    nearest = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    blur_nearest = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_NEAREST)
    area = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # піраміда: pyrDown = гаус 5x5 + кожен другий піксель, два рівні = у 4 рази
    pyramid = cv2.pyrDown(cv2.pyrDown(image))
    return nearest, blur_nearest, area, pyramid


# зонна пластинка, локальна частота f = r / N
N = 1024
yy, xx = np.mgrid[0:N, 0:N] - N / 2
r = np.sqrt(xx ** 2 + yy ** 2)
zone = (127.5 + 127.5 * np.cos(np.pi * r ** 2 / N)).astype(np.uint8)

# після зменшення в 4 рази Найквіст = 0.125; вище має бути сіре, std там = муар
r_small = r[::4, ::4]
zone_moire = (r_small / N > 0.15) & (r_small / N < 0.48)
zone_low = r_small / N < 0.075
low_std_original = zone[r < 0.075 * N].std()

rows = []
names = ["INTER_NEAREST", "GaussianBlur + NEAREST", "INTER_AREA", "pyrDown x2"]
for name, small in zip(names, downscale_4_ways(zone)):
    moire = small[zone_moire].std()
    kept = small[zone_low].std() / low_std_original * 100
    print(f"{name:24s} муар (std) = {moire:.1f}   збережений контраст = {kept:.1f} %")
    rows.append([name, round(moire, 1), round(kept, 1)])
save_csv("c1_aliasing.csv", ["метод", "std у зоні f > f_Найквіста", "збережений контраст низьких частот, %"], rows)

# рисунок
fig, ax = plt.subplots(2, 6, figsize=(24, 9))
for row, (title, image) in enumerate([("зонна пластинка", zone), ("вишивка (texture.jpg)", texture)]):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    # без antialiased муар видно вже на самому оригіналі
    ax[row, 0].imshow(image, cmap="gray", interpolation="antialiased")
    ax[row, 0].set_title(f"{title}\n{w}×{h}")
    for i, small in enumerate(downscale_4_ways(image)):
        ax[row, i + 1].imshow(small, cmap="gray", interpolation="nearest", extent=(0, w, h, 0))
        ax[row, i + 1].set_title(f"{names[i]}\n{small.shape[1]}×{small.shape[0]}")
    # шматок 1:1
    if row == 0:
        piece = image[448:576, 806:934]
    else:
        piece = image[936:1064, 686:814]
    ax[row, 5].imshow(piece, cmap="gray", interpolation="nearest")
    ax[row, 5].set_title("фрагмент оригіналу 128×128")
    for a in ax[row]:
        a.axis("off")
plt.savefig(os.path.join(FIG, "c1_aliasing.png"), dpi=150, bbox_inches="tight")
plt.close()


# Крок 13. Інтерполяція
print("\nКрок 13")
methods = [["INTER_NEAREST", cv2.INTER_NEAREST],
           ["INTER_LINEAR", cv2.INTER_LINEAR],
           ["INTER_CUBIC", cv2.INTER_CUBIC],
           ["INTER_LANCZOS4", cv2.INTER_LANCZOS4]]

# шматок з горизонтом (на самому лісі й на вишивці всі методи виходять однакові)
crop = photo[500:1012, 760:1272]
small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

fig, ax = plt.subplots(2, 3, figsize=(13, 9))
ax = ax.ravel()
ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax[0].set_title("оригінальний фрагмент 512×512")
ax[1].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), interpolation="nearest")
ax[1].set_title("зменшений у 8 разів: 64×64")

# ще 5 шматків уздовж горизонту - для розкиду
x_list = [0, 380, 760, 1140, 1536]

rows = []
for i, (name, flag) in enumerate(methods):
    big = cv2.resize(small, (512, 512), interpolation=flag)
    p = psnr(crop, big)
    s = structural_similarity(crop, big, channel_axis=2)
    ax[i + 2].imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    ax[i + 2].set_title(f"{name}\nPSNR {p:.2f} дБ, SSIM {s:.3f}")

    psnr_list = []
    ssim_list = []
    for x0 in x_list:
        part = photo[500:1012, x0:x0 + 512]
        part_small = cv2.resize(part, (64, 64), interpolation=cv2.INTER_AREA)
        part_big = cv2.resize(part_small, (512, 512), interpolation=flag)
        psnr_list.append(psnr(part, part_big))
        ssim_list.append(structural_similarity(part, part_big, channel_axis=2))

    # час: 20 замірів
    times = []
    for _ in range(20):
        start = time.perf_counter()
        cv2.resize(small, (512, 512), interpolation=flag)
        times.append((time.perf_counter() - start) * 1000)

    print(f"{name:15s} PSNR = {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} дБ  "
          f"SSIM = {np.mean(ssim_list):.3f} ± {np.std(ssim_list):.3f}  "
          f"час = {np.median(times):.3f} мс (std {np.std(times):.3f})")
    rows.append([name, round(np.mean(psnr_list), 2), round(np.std(psnr_list), 2),
                 round(np.mean(ssim_list), 3), round(np.std(ssim_list), 3),
                 round(np.median(times), 3), round(np.std(times), 3)])
for a in ax:
    a.axis("off")
plt.savefig(os.path.join(FIG, "c2_interpolation.png"), dpi=150, bbox_inches="tight")
plt.close()

save_csv("c2_interpolation.csv", ["метод", "PSNR середнє, дБ", "PSNR std", "SSIM середнє", "SSIM std",
                                    "час медіана, мс", "час std"], rows)


# Крок 14. Гамма
print("\nКрок 14")
gammas = [0.5, 1.0, 2.2]

# out = 255 * (in / 255) ^ g
x = np.arange(256)
plt.figure(figsize=(5.5, 5))
for g in gammas:
    plt.plot(x, 255 * (x / 255) ** g, label=f"γ = {g}")
plt.xlabel("вхідне значення, рівні 0-255")
plt.ylabel("вихідне значення, рівні 0-255")
plt.title("Передавальні характеристики")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FIG, "c3_gamma_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for i, g in enumerate(gammas):
    table = (255 * (np.arange(256) / 255) ** g).astype(np.uint8)
    corrected = cv2.LUT(photo, table)
    ax[i].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    ax[i].set_title(f"γ = {g}")
    ax[i].axis("off")
plt.savefig(os.path.join(FIG, "c3_gamma_images.png"), dpi=150, bbox_inches="tight")
plt.close()

# освітленість L = (v / 255) ^ 2.2
rows = []
for v1, v2 in [[0, 255], [50, 200], [100, 150], [0, 100]]:
    mean_codes = (v1 + v2) / 2
    mean_light = ((v1 / 255) ** 2.2 + (v2 / 255) ** 2.2) / 2
    mean_light_code = 255 * mean_light ** (1 / 2.2)
    print(f"{v1} і {v2}: середнє кодів = {mean_codes}, середнє освітленостей = {mean_light_code:.1f}")
    rows.append([v1, v2, mean_codes, round(mean_light_code, 1)])
save_csv("c3_gamma_average.csv", ["v1", "v2", "середнє кодів", "середнє освітленостей (у кодах)"], rows)

# смуги 0/255 і сірі 128, 186 (дивитись у масштабі 100 %)
demo = np.zeros((200, 900), dtype=np.uint8)
demo[:, 0:300:2] = 255
demo[:, 300:600] = 128
demo[:, 600:900] = 186
cv2.imwrite(os.path.join(FIG, "c3_gamma_stripes.png"), demo)


# Сірий світ
print("\nСірий світ")
lamp = cv2.imread(os.path.join(DATA, "object_light2.jpg"))
means = lamp.reshape(-1, 3).mean(axis=0)
gain = means.mean() / means
balanced = np.clip(lamp * gain, 0, 255).astype(np.uint8)
means_after = balanced.reshape(-1, 3).mean(axis=0)
print("до:", means.round(1), "R/B =", round(means[2] / means[0], 2))
print("після:", means_after.round(1), "R/B =", round(means_after[2] / means_after[0], 2))
save_csv("c3_gray_world.csv", ["", "середнє B", "середнє G", "середнє R", "R/B"], [
    ["до"] + list(means.round(1)) + [round(means[2] / means[0], 2)],
    ["після"] + list(means_after.round(1)) + [round(means_after[2] / means_after[0], 2)],
])

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(cv2.cvtColor(lamp, cv2.COLOR_BGR2RGB))
ax[0].set_title("кадр під лампою")
ax[1].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
ax[1].set_title("після «сірого світу»")
ax[0].axis("off")
ax[1].axis("off")
plt.savefig(os.path.join(FIG, "c3_gray_world.png"), dpi=150, bbox_inches="tight")
plt.close()
