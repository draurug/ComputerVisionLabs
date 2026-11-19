# Lab 6, part D: MOG2 background subtraction, cleaning, blobs, counting people that cross a line

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker import MIN_AREA, Tracker, blobs, masks
from utils import FIG, frame_with_label, hstack, read_clip, save_csv, writer

np.random.seed(42)

LINE_Y = 300        # px, horizontal line across clip B (the right part of the road, where people walk down)
SHOW = 60           # frame of clip B shown step by step
MANUAL = {"up": 2, "down": 2}   # step 15, counted by looking through the clip (see README)


def count_crossings(frames, kalman):
    # Step 14. a track is counted when its matched centroid goes from one side of the line to the other;
    # every track id is counted only once
    tracker = Tracker(kalman)
    last_y = {}
    counted = set()
    events = []
    out = writer(f"counting_{'kalman' if kalman else 'simple'}.avi", frames[0].shape) if not kalman else None
    for k, (_, mask) in enumerate(masks(frames)):
        centres, boxes = blobs(mask)
        tracks = tracker.step(centres)
        for t in tracks:
            if t.missed:
                continue
            y = float(t.pos[1])
            if t.id in last_y and t.id not in counted:
                if last_y[t.id] < LINE_Y <= y:
                    events.append([k, t.id, "down", round(float(t.pos[0]))])
                    counted.add(t.id)
                elif last_y[t.id] >= LINE_Y > y:
                    events.append([k, t.id, "up", round(float(t.pos[0]))])
                    counted.add(t.id)
            last_y[t.id] = y
        if out is not None:
            pic = frames[k].copy()
            cv2.line(pic, (0, LINE_Y), (pic.shape[1], LINE_Y), (0, 0, 255), 2)
            for x, y, w, h in boxes:
                cv2.rectangle(pic, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            for t in tracks:
                color = (0, 255, 255) if t.missed else (255, 255, 0)
                cv2.circle(pic, tuple(int(v) for v in t.pos), 4, color, -1)
                cv2.putText(pic, str(t.id), (int(t.pos[0]) + 5, int(t.pos[1]) - 5), 0, 0.5, color, 2)
            ups = sum(e[2] == "up" for e in events)
            cv2.putText(pic, f"up {ups}  down {len(events) - ups}", (10, 25), 0, 0.8, (0, 0, 255), 2)
            out.write(pic)
    if out is not None:
        out.release()
    return events


frames = read_clip("B")

# Steps 11-13 on one frame: raw mask, shadows removed, after opening and closing, kept blobs
areas = []
for k, (raw, mask) in enumerate(masks(frames)):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if k >= 20:
        # the first frames are skipped: the model is still learning the background
        areas.extend(stats[1:, cv2.CC_STAT_AREA])
    if k == SHOW:
        no_shadow = np.where(raw == 255, 255, 0).astype(np.uint8)
        _, boxes = blobs(mask)
        kept = frames[k].copy()
        for x, y, w, h in boxes:
            cv2.rectangle(kept, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        small = lambda im: cv2.resize(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), None,
                                      fx=0.5, fy=0.5)
        top = hstack([small(frame_with_label(frames[k], "frame")),
                      small(frame_with_label(cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), "MOG2: 255 object, 127 shadow"))])
        bottom = hstack([small(frame_with_label(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "no shadow, open 3, close 5")),
                         small(frame_with_label(kept, f"blobs >= {MIN_AREA} px"))])
        cv2.imwrite(os.path.join(FIG, "mog2_steps.jpg"), np.vstack([top, bottom]))
        print(f"frame {SHOW}: shadow pixels {np.mean(raw == 127) * 100:.2f} %, object pixels "
              f"{np.mean(raw == 255) * 100:.2f} %, blobs before the area filter {n - 1}, after {len(boxes)}")

areas = np.array(areas)
print(f"blob areas after frame 20: {len(areas)} blobs, median {np.median(areas):.0f} px, "
      f"below {MIN_AREA} px: {np.mean(areas < MIN_AREA) * 100:.1f} %")

plt.figure(figsize=(6, 3.5))
plt.hist(areas, bins=np.logspace(0, np.log10(areas.max() + 1), 50))
plt.axvline(MIN_AREA, color="r", label=f"threshold {MIN_AREA} px")
plt.xscale("log")
plt.xlabel("blob area, px")
plt.ylabel("blobs (frames 20-199)")
plt.title("Connected components after opening and closing, clip B")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FIG, "blob_areas.png"), dpi=150, bbox_inches="tight")
plt.close()

# Steps 14-15
rows = []
all_events = []
for kalman in (False, True):
    label = "Kalman" if kalman else "simple"
    events = count_crossings(frames, kalman)
    up = sum(e[2] == "up" for e in events)
    rows.append([label, up, len(events) - up, len(events)])
    all_events += [[label] + e for e in events]
    print(f"{label} tracker: up {up}, down {len(events) - up}, events {events}")
rows.append(["manual", MANUAL["up"], MANUAL["down"], MANUAL["up"] + MANUAL["down"]])

save_csv("line_count.csv", ["method", "up", "down", "total"], rows)
save_csv("line_events.csv", ["tracker", "frame", "track id", "direction", "x, px"], all_events)
save_csv("blobs.csv", ["blobs (frames 20-199)", "median area, px", "share below threshold", "threshold, px"],
         [[len(areas), int(np.median(areas)), round(np.mean(areas < MIN_AREA), 3), MIN_AREA]])
