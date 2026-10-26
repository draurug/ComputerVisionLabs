# Lab 1, part C, steps 12-14

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


# Step 12. Aliasing
print("Step 12")


def downscale_4_ways(image):
    w, h = image.shape[1] // 4, image.shape[0] // 4
    nearest = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    blur_nearest = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_NEAREST)
    area = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # pyrDown = 5x5 gaussian + every second pixel, two levels = 4 times smaller
    pyramid = cv2.pyrDown(cv2.pyrDown(image))
    return nearest, blur_nearest, area, pyramid


# zone plate, local frequency f = r / N
N = 1024
yy, xx = np.mgrid[0:N, 0:N] - N / 2
r = np.sqrt(xx ** 2 + yy ** 2)
zone = (127.5 + 127.5 * np.cos(np.pi * r ** 2 / N)).astype(np.uint8)

# after 4x downscaling Nyquist is 0.125; above it the image should be flat gray, std there = moire
r_small = r[::4, ::4]
zone_moire = (r_small / N > 0.15) & (r_small / N < 0.48)
zone_low = r_small / N < 0.075
low_std_original = zone[r < 0.075 * N].std()

rows = []
names = ["INTER_NEAREST", "GaussianBlur + NEAREST", "INTER_AREA", "pyrDown x2"]
for name, small in zip(names, downscale_4_ways(zone)):
    moire = small[zone_moire].std()
    kept = small[zone_low].std() / low_std_original * 100
    print(f"{name:24s} moire (std) = {moire:.1f}   contrast kept = {kept:.1f} %")
    rows.append([name, round(moire, 1), round(kept, 1)])
save_csv("c1_aliasing.csv", ["method", "std where f > Nyquist", "low frequency contrast kept, %"], rows)

# figure
fig, ax = plt.subplots(2, 6, figsize=(24, 9))
for row, (title, image) in enumerate([("zone plate", zone), ("embroidery (texture.jpg)", texture)]):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    # without "antialiased" the moire shows up on the original itself
    ax[row, 0].imshow(image, cmap="gray", interpolation="antialiased")
    ax[row, 0].set_title(f"{title}\n{w}×{h}")
    for i, small in enumerate(downscale_4_ways(image)):
        ax[row, i + 1].imshow(small, cmap="gray", interpolation="nearest", extent=(0, w, h, 0))
        ax[row, i + 1].set_title(f"{names[i]}\n{small.shape[1]}×{small.shape[0]}")
    # 1:1 crop
    if row == 0:
        piece = image[448:576, 806:934]
    else:
        piece = image[936:1064, 686:814]
    ax[row, 5].imshow(piece, cmap="gray", interpolation="nearest")
    ax[row, 5].set_title("original crop 128×128")
    for a in ax[row]:
        a.axis("off")
plt.savefig(os.path.join(FIG, "c1_aliasing.png"), dpi=150, bbox_inches="tight")
plt.close()


# Step 13. Interpolation
print("\nStep 13")
methods = [["INTER_NEAREST", cv2.INTER_NEAREST],
           ["INTER_LINEAR", cv2.INTER_LINEAR],
           ["INTER_CUBIC", cv2.INTER_CUBIC],
           ["INTER_LANCZOS4", cv2.INTER_LANCZOS4]]

# crop with the horizon (on the forest or the embroidery all methods look the same)
crop = photo[500:1012, 760:1272]
small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

fig, ax = plt.subplots(2, 3, figsize=(13, 9))
ax = ax.ravel()
ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax[0].set_title("original crop 512×512")
ax[1].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), interpolation="nearest")
ax[1].set_title("downscaled 8 times: 64×64")

# 5 more crops along the horizon to get the spread
x_list = [0, 380, 760, 1140, 1536]

rows = []
for i, (name, flag) in enumerate(methods):
    big = cv2.resize(small, (512, 512), interpolation=flag)
    p = psnr(crop, big)
    s = structural_similarity(crop, big, channel_axis=2)
    ax[i + 2].imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    ax[i + 2].set_title(f"{name}\nPSNR {p:.2f} dB, SSIM {s:.3f}")

    psnr_list = []
    ssim_list = []
    for x0 in x_list:
        part = photo[500:1012, x0:x0 + 512]
        part_small = cv2.resize(part, (64, 64), interpolation=cv2.INTER_AREA)
        part_big = cv2.resize(part_small, (512, 512), interpolation=flag)
        psnr_list.append(psnr(part, part_big))
        ssim_list.append(structural_similarity(part, part_big, channel_axis=2))

    # time: 20 runs
    times = []
    for _ in range(20):
        start = time.perf_counter()
        cv2.resize(small, (512, 512), interpolation=flag)
        times.append((time.perf_counter() - start) * 1000)

    print(f"{name:15s} PSNR = {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB  "
          f"SSIM = {np.mean(ssim_list):.3f} ± {np.std(ssim_list):.3f}  "
          f"time = {np.median(times):.3f} ms (std {np.std(times):.3f})")
    rows.append([name, round(np.mean(psnr_list), 2), round(np.std(psnr_list), 2),
                 round(np.mean(ssim_list), 3), round(np.std(ssim_list), 3),
                 round(np.median(times), 3), round(np.std(times), 3)])
for a in ax:
    a.axis("off")
plt.savefig(os.path.join(FIG, "c2_interpolation.png"), dpi=150, bbox_inches="tight")
plt.close()

save_csv("c2_interpolation.csv", ["method", "PSNR mean, dB", "PSNR std", "SSIM mean", "SSIM std",
                                    "time median, ms", "time std"], rows)


# Step 14. Gamma
print("\nStep 14")
gammas = [0.5, 1.0, 2.2]

# out = 255 * (in / 255) ^ g
x = np.arange(256)
plt.figure(figsize=(5.5, 5))
for g in gammas:
    plt.plot(x, 255 * (x / 255) ** g, label=f"γ = {g}")
plt.xlabel("input value, 0-255")
plt.ylabel("output value, 0-255")
plt.title("Transfer curves")
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

# linear light L = (v / 255) ^ 2.2
rows = []
for v1, v2 in [[0, 255], [50, 200], [100, 150], [0, 100]]:
    mean_codes = (v1 + v2) / 2
    mean_light = ((v1 / 255) ** 2.2 + (v2 / 255) ** 2.2) / 2
    mean_light_code = 255 * mean_light ** (1 / 2.2)
    print(f"{v1} and {v2}: mean of codes = {mean_codes}, mean of linear light = {mean_light_code:.1f}")
    rows.append([v1, v2, mean_codes, round(mean_light_code, 1)])
save_csv("c3_gamma_average.csv", ["v1", "v2", "mean of codes", "mean of linear light (as code)"], rows)

# 0/255 stripes next to gray 128 and 186 (look at it at 100% zoom)
demo = np.zeros((200, 900), dtype=np.uint8)
demo[:, 0:300:2] = 255
demo[:, 300:600] = 128
demo[:, 600:900] = 186
cv2.imwrite(os.path.join(FIG, "c3_gamma_stripes.png"), demo)


# Gray world
print("\nGray world")
lamp = cv2.imread(os.path.join(DATA, "object_light2.jpg"))
means = lamp.reshape(-1, 3).mean(axis=0)
gain = means.mean() / means
balanced = np.clip(lamp * gain, 0, 255).astype(np.uint8)
means_after = balanced.reshape(-1, 3).mean(axis=0)
print("before:", means.round(1), "R/B =", round(means[2] / means[0], 2))
print("after:", means_after.round(1), "R/B =", round(means_after[2] / means_after[0], 2))
save_csv("c3_gray_world.csv", ["", "mean B", "mean G", "mean R", "R/B"], [
    ["before"] + list(means.round(1)) + [round(means[2] / means[0], 2)],
    ["after"] + list(means_after.round(1)) + [round(means_after[2] / means_after[0], 2)],
])

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(cv2.cvtColor(lamp, cv2.COLOR_BGR2RGB))
ax[0].set_title("lamp light")
ax[1].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
ax[1].set_title("after gray world")
ax[0].axis("off")
ax[1].axis("off")
plt.savefig(os.path.join(FIG, "c3_gray_world.png"), dpi=150, bbox_inches="tight")
plt.close()
