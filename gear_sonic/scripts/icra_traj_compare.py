#!/usr/bin/env python3
"""
icra_traj_compare.py — compare the robot's commanded walk path against the human demo.

Reads:
  - the monitor JSONL (base_trans_target = the planner's integrated commanded base path)
  - the ICRA dataset episode (head world XY = the human trajectory)

Aligns both to start at the origin with the initial heading along +X (the robot's
frame-0 facing), then reports path length, net displacement, the robot/human ratios,
and a resampled side-by-side so you can eyeball the shape.

NOTE: in sim the deploy does not feed back base position (base_trans_measured is
constant), so we compare the *commanded* robot path. That measures how well our
locomotion mapping reproduces the human trajectory.

Usage:
    python gear_sonic/scripts/icra_traj_compare.py --episode 0
    python gear_sonic/scripts/icra_traj_compare.py --episode 0 --log graphify-out/icra_sonic_monitor.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as sRot


def robot_path(log_path):
    rows = [json.loads(l) for l in open(log_path)]
    pts = np.array([r["base_trans_target"][:2] for r in rows if "base_trans_target" in r])
    if len(pts) == 0:
        raise SystemExit("No base_trans_target in log — run the monitor during a walk first.")
    keep = [0] + [i for i in range(1, len(pts)) if not np.allclose(pts[i], pts[i - 1])]
    pts = pts[keep]
    return pts - pts[0]


def human_path(icra_root, episode):
    df = pd.read_parquet(Path(icra_root) / "data" / "chunk-000" / "file-000.parquet")
    ep = df[df["episode_index"] == episode].sort_values("frame_index")
    H = np.stack(ep["observation.pose.head"].values)
    look0 = sRot.from_quat(H[0, 3:7]).apply([0, 0, 1])
    yaw0 = np.arctan2(look0[1], look0[0])
    xy = H[:, :2] - H[0, :2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return (np.array([[c, -s], [s, c]]) @ xy.T).T


def path_len(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def resample(P, n=8):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if d[-1] < 1e-6:
        return np.zeros((n, 2))
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--icra-root", default=str(Path.home() / "icra-dataset"))
    ap.add_argument("--log", default="graphify-out/icra_sonic_monitor.jsonl")
    args = ap.parse_args()

    rob = robot_path(args.log)
    hum = human_path(args.icra_root, args.episode)
    np.set_printoptions(precision=3, suppress=True)

    rl, hl = path_len(rob), path_len(hum)
    rn, hn = float(np.linalg.norm(rob[-1])), float(np.linalg.norm(hum[-1]))
    rh = np.degrees(np.arctan2(rob[-1, 1], rob[-1, 0]))
    hh = np.degrees(np.arctan2(hum[-1, 1], hum[-1, 0]))

    print(f"ROBOT (commanded): path {rl:.2f} m | net {rn:.2f} m | end-heading {rh:+.0f}° | end {rob[-1]}")
    print(f"HUMAN (head):      path {hl:.2f} m | net {hn:.2f} m | end-heading {hh:+.0f}° | end {hum[-1]}")
    print(f"RATIO robot/human: path {rl/max(1e-6,hl):.2f}× | net {rn/max(1e-6,hn):.2f}×  "
          f"(want ~1.0)   heading gap {rh-hh:+.0f}°")
    print("\nrobot resampled:\n", resample(rob))
    print("human resampled:\n", resample(hum))


if __name__ == "__main__":
    main()
