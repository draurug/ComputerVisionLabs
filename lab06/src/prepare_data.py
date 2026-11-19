# Lab 6, part A: get vtest.avi, describe the two clips, step 2 brightness check

import os
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import CLIPS, FIG, FPS, VIDEO, gray, read_clip, save_csv

URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"

if not os.path.exists(VIDEO):
    print("downloading", URL)
    urllib.request.urlretrieve(URL, VIDEO)

video = cv2.VideoCapture(VIDEO)
print(f"vtest.avi: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames, {video.get(cv2.CAP_PROP_FPS):.0f} fps, "
      f"{int(video.get(3))}x{int(video.get(4))}")
video.release()

# Step 2. Exposure: mean brightness of every frame should stay the same.
# People walking through the frame change it a little, so the background alone is checked too:
# pixels that are not far from the median frame.
rows = []
plt.figure(figsize=(7, 3.5))
for name, clip in CLIPS.items():
    frames = [gray(f).astype(np.float32) for f in read_clip(name)]
    median = np.median(frames, axis=0)
    mean_all = np.array([f.mean() for f in frames])
    mean_bg = np.array([f[np.abs(f - median) < 25].mean() for f in frames])
    h, w = frames[0].shape
    rows.append([name, clip["first"], clip["last"], len(frames), round(len(frames) / FPS, 1),
                 f"{w}x{h}", str(clip["roi"]),
                 round(mean_all.mean(), 2), round(mean_all.std(), 2), round(mean_all.max() - mean_all.min(), 2),
                 round(mean_bg.mean(), 2), round(mean_bg.std(), 2), round(mean_bg.max() - mean_bg.min(), 2)])
    print(f"clip {name}: frames {clip['first']}-{clip['last']} ({len(frames)}, {len(frames) / FPS:.1f} s), "
          f"{w}x{h}, brightness {mean_all.mean():.1f} +- {mean_all.std():.2f}, "
          f"background {mean_bg.mean():.1f} +- {mean_bg.std():.2f}")
    plt.plot(np.arange(len(frames)), mean_all, label=f"clip {name}, whole frame")
    plt.plot(np.arange(len(frames)), mean_bg, "--", label=f"clip {name}, background")

plt.xlabel("frame of the clip")
plt.ylabel("mean brightness, 0-255")
plt.title("Step 2: brightness over time")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.savefig(os.path.join(FIG, "brightness.png"), dpi=150, bbox_inches="tight")
plt.close()

save_csv("clips.csv", ["clip", "first frame", "last frame", "frames", "duration, s", "size", "roi (x, y, w, h)",
                       "mean brightness", "std", "max - min",
                       "background brightness", "background std", "background max - min"], rows)
