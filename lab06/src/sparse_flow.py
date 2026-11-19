# Lab 6, part B: Lucas-Kanade on goodFeaturesToTrack points, forward-backward check, trajectories

import os
from collections import deque

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import FB_MAX, FEATURES, FIG, LK, frame_with_label, gray, hstack, read_clip, save_csv, writer

np.random.seed(42)

TAIL = 20         # last positions drawn, step 7
MOVING = 5.0      # px, a point that went farther than this from its start is on a moving object
SHOW = {"A": 60, "B": 35}   # frame of the clip saved as a picture, people are walking there


def track(frames, fb_check, max_level=2):
    # returns positions and alive masks for every frame
    params = dict(LK, maxLevel=max_level)
    prev = gray(frames[0])
    pts = cv2.goodFeaturesToTrack(prev, **FEATURES).reshape(-1, 2)
    alive = np.ones(len(pts), bool)
    positions, masks = [pts.copy()], [alive.copy()]
    for frame in frames[1:]:
        cur = gray(frame)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts.reshape(-1, 1, 2), None, **params)
        nxt = nxt.reshape(-1, 2)
        ok = st.ravel() == 1
        if fb_check:
            # Step 5. track back from the new positions, the point must come back within 1 px
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt.reshape(-1, 1, 2), None, **params)
            err = np.linalg.norm(back.reshape(-1, 2) - pts, axis=1)
            ok &= (st_b.ravel() == 1) & (err <= FB_MAX)
        # a lost point stays lost and keeps its last position
        alive = alive & ok
        pts = np.where(alive[:, None], nxt, pts)
        positions.append(pts.copy())
        masks.append(alive.copy())
        prev = cur
    return np.array(positions), np.array(masks)


def draw_tracks(frames, positions, masks, video, show):
    # Step 7. last 20 positions of every live point, written to a video; returns the frame number show
    out = writer(video, frames[0].shape)
    colors = np.random.randint(0, 255, (positions.shape[1], 3))
    tails = [deque(maxlen=TAIL) for _ in range(positions.shape[1])]
    for k, frame in enumerate(frames):
        pic = frame.copy()
        for i, tail in enumerate(tails):
            if not masks[k][i]:
                tail.clear()
                continue
            tail.append(tuple(int(v) for v in positions[k][i]))
            color = tuple(int(c) for c in colors[i])
            for a, b in zip(tail, list(tail)[1:]):
                cv2.line(pic, a, b, color, 2)
            cv2.circle(pic, tail[-1], 3, color, -1)
        out.write(pic)
        if k == show:
            shown = pic
    out.release()
    return shown


rows = []
curves = {}
for name in ["A", "B"]:
    frames = read_clip(name)
    pictures = []
    for label, fb in [("no filter", False), ("forward-backward <= 1 px", True)]:
        positions, masks = track(frames, fb)
        counts = masks.sum(axis=1)
        curves[(name, fb)] = counts
        # points that really moved: final position far from the start (the rest sit on the static background)
        moved = np.linalg.norm(positions[-1] - positions[0], axis=1) > MOVING
        rows.append([name, label, counts[0], counts[-1], round(counts[-1] / counts[0], 3),
                     int(moved.sum()), int((moved & masks[-1]).sum())])
        print(f"clip {name}, {label}: {counts[0]} -> {counts[-1]} ({counts[-1] / counts[0] * 100:.1f} %), "
              f"moved > {MOVING} px: {moved.sum()}, of them alive at the end: {(moved & masks[-1]).sum()}")
        tag = "filtered" if fb else "raw"
        pic = draw_tracks(frames, positions, masks, f"trajectories_{name}_{tag}.avi", SHOW[name])
        pictures.append(frame_with_label(pic, f"clip {name}, frame {SHOW[name]}, {label}"))
    cv2.imwrite(os.path.join(FIG, f"trajectories_{name}.jpg"), hstack(pictures))
    # the same with four levels, to see if the pyramid depth matters here
    _, masks4 = track(frames, True, max_level=3)
    print(f"  maxLevel = 3 (four levels), with the check: {masks4[-1].sum()} alive at the end")
    rows.append([name, "forward-backward, maxLevel = 3", masks4[0].sum(), masks4[-1].sum(),
                 round(masks4[-1].sum() / masks4[0].sum(), 3), "", ""])

# Step 6. number of live points per frame, two curves per clip
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, name in zip(axes, ["A", "B"]):
    ax.plot(curves[(name, False)], label="no filter")
    ax.plot(curves[(name, True)], label="forward-backward <= 1 px")
    ax.set_xlabel("frame of the clip")
    ax.set_ylabel("live points")
    ax.set_title(f"clip {name}")
    ax.set_ylim(0, curves[(name, False)][0] * 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "sparse_survival.png"), dpi=150, bbox_inches="tight")
plt.close()

save_csv("sparse_survival.csv", ["clip", "filter", "points at start", "points at end", "share",
                                 "moved > 5 px", "moved and alive at end"], rows)
