#!/usr/bin/env python3
"""
icra_sonic_monitor.py — passive observer of the ICRA→Sonic sim pipeline.

Subscribes to BOTH ends of the loop without injecting anything, so you can see
the gap between what icra_to_groot.py commands and what the Sonic WBC actually
does in MuJoCo:

  INPUT  — ``planner`` topic (port 5556, from icra_to_groot.py):
             vr_3pt_position/orientation, mode, movement, facing, speed, height
  OUTPUT — ``g1_debug`` topic (port 5557, from the C++ deploy):
             body_q → FK of the 3 vr-target links (pelvis-relative), base tilt

The three WBC vr-target bodies are, per config/manager_env/commands/terms/motion.yaml:
    ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
We FK exactly those from the measured joints and print commanded-vs-actual side by
side, plus a per-link position error. Everything is also appended to a JSONL log.

Run (in the data-collection venv, which has the robot model):
    source .venv_data_collection/bin/activate
    python gear_sonic/scripts/icra_sonic_monitor.py

Then start deploy + icra_to_groot in the other terminals as usual. Ctrl-C to stop.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE  # 1280

# Reuse the exporter's robust unpackers so we match the wire format exactly.
from gear_sonic.scripts.run_data_exporter import unpack_pose_message
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    _unpack_msgpack_zmq,
    _convert_lists_to_numpy,
    STATE_ZMQ_TOPIC,
)

# WBC vr_3point_body order (motion.yaml:49). NOTE: 3rd point is torso_link, not "neck".
VR_TARGET_FRAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
POINT_LABELS = ["LW", "RW", "TORSO"]
# vr_3point_body_offset (motion.yaml:49): the WBC tracks link_frame + offset (LOCAL frame),
# NOT the bare link. Must apply these or commanded-vs-actual error is overstated.
VR_TARGET_OFFSETS = {
    "left_wrist_yaw_link":  np.array([0.18, -0.025, 0.0]),
    "right_wrist_yaw_link": np.array([0.18, +0.025, 0.0]),
    "torso_link":           np.array([0.0, 0.0, 0.35]),
}


def load_robot_model():
    """Load the G1 model for FK. Returns None if unavailable (then we skip FK)."""
    try:
        from gear_sonic.data.features_sonic_vla import get_g1_robot_model
        rm = get_g1_robot_model()
        print("[Monitor] Robot model loaded — FK of actual pose enabled.")
        return rm
    except Exception as e:
        print(f"[Monitor] Robot model unavailable ({e}); logging raw state only.")
        return None


def fk_pelvis_relative(rm, body_q, left_hand_q, right_hand_q):
    """FK the 3 vr-target frames, expressed pelvis-relative (robot frame).

    get_configuration_from_actuated_joints reconstructs whole_q with the floating
    base at the origin, so frame_placement is already pelvis-relative.
    Returns dict frame_name -> (pos(3,), quat_wxyz(4,)) or None on failure.
    """
    try:
        whole_q = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q,
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
        rm.cache_forward_kinematics(whole_q)
        out = {}
        for name in VR_TARGET_FRAMES:
            p = rm.frame_placement(name)
            t = np.asarray(p.translation, dtype=np.float64).copy()
            Rm = np.asarray(p.rotation, dtype=np.float64)
            # Apply the WBC's local-frame offset so we compare against the point the
            # controller actually tracks (motion.yaml vr_3point_body_offset).
            t = t + Rm @ VR_TARGET_OFFSETS[name]
            out[name] = (t, sRot.from_matrix(Rm).as_quat(scalar_first=True))
        return out
    except Exception as e:
        print(f"[Monitor] FK failed: {e}")
        return None


def base_euler_deg(base_quat_wxyz):
    """Roll/pitch/yaw of the base in degrees (0,0,0 = upright, facing +X). [w,x,y,z]."""
    try:
        q = np.asarray(base_quat_wxyz, dtype=np.float64)
        r = sRot.from_quat([q[1], q[2], q[3], q[0]])  # scipy wants scalar-last
        roll, pitch, yaw = r.as_euler("xyz", degrees=True)
        return float(roll), float(pitch), float(yaw)
    except Exception:
        return None, None, None


def facing_yaw_deg(facing):
    """Commanded heading: yaw of the facing vector in the XY plane, degrees."""
    try:
        return float(np.degrees(np.arctan2(facing[1], facing[0])))
    except Exception:
        return None


def parse_planner(raw):
    """Decode a 'planner' message into a flat dict of the fields we care about."""
    d = unpack_pose_message(raw, topic="planner")
    out = {}
    if "vr_position" in d and d["vr_position"].size == 9:
        out["vr_pos"] = d["vr_position"].flatten().astype(np.float64)
    if "vr_orientation" in d and d["vr_orientation"].size == 12:
        out["vr_orn"] = d["vr_orientation"].flatten().astype(np.float64)
    out["mode"] = int(d["mode"].flat[0]) if "mode" in d else None
    out["movement"] = d["movement"].flatten().tolist() if "movement" in d else None
    out["facing"] = d["facing"].flatten().tolist() if "facing" in d else None
    out["speed"] = float(d["speed"].flat[0]) if "speed" in d else None
    out["height"] = float(d["height"].flat[0]) if "height" in d else None
    out["lgrip"] = float(d["left_gripper_cmd"].flat[0]) if "left_gripper_cmd" in d else None
    out["rgrip"] = float(d["right_gripper_cmd"].flat[0]) if "right_gripper_cmd" in d else None
    return out


def fmt3(v):
    return "[" + " ".join(f"{x:+.3f}" for x in v[:3]) + "]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--planner-port", type=int, default=5556)
    ap.add_argument("--state-port", type=int, default=5557)
    ap.add_argument("--rate", type=float, default=5.0, help="Console print rate (Hz)")
    ap.add_argument("--log-rate", type=float, default=30.0, help="JSONL log rate (Hz)")
    ap.add_argument("--log", default="graphify-out/icra_sonic_monitor.jsonl",
                    help="JSONL output path (set '' to disable)")
    args = ap.parse_args()

    rm = load_robot_model()

    ctx = zmq.Context()
    planner_sub = ctx.socket(zmq.SUB)
    planner_sub.connect(f"tcp://{args.host}:{args.planner_port}")
    planner_sub.setsockopt_string(zmq.SUBSCRIBE, "planner")
    planner_sub.setsockopt(zmq.CONFLATE, 0)
    planner_sub.setsockopt(zmq.RCVHWM, 50)

    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(f"tcp://{args.host}:{args.state_port}")
    state_sub.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    state_sub.setsockopt(zmq.CONFLATE, 1)

    print(f"[Monitor] planner SUB tcp://{args.host}:{args.planner_port}")
    print(f"[Monitor] state   SUB tcp://{args.host}:{args.state_port} (topic {STATE_ZMQ_TOPIC})")
    print("[Monitor] Waiting for messages... (Ctrl-C to stop)\n")

    logf = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        logf = open(args.log, "w")  # truncate: one fresh file per run
        print(f"[Monitor] Logging JSONL to {args.log} (fresh, ~{args.log_rate:.0f} Hz)\n")

    latest_planner = None
    latest_state = None
    state_keys_printed = False
    last_print = 0.0
    last_log = 0.0
    print_period = 1.0 / args.rate
    log_period = 1.0 / args.log_rate if args.log_rate > 0 else 0.0

    try:
        while True:
            # drain planner (keep most recent)
            while True:
                try:
                    raw = planner_sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    latest_planner = parse_planner(raw)
                except Exception as e:
                    print(f"[Monitor] planner parse error: {e}")

            # latest state (conflated)
            try:
                raw = state_sub.recv(zmq.NOBLOCK)
                msg = _convert_lists_to_numpy(_unpack_msgpack_zmq(raw, STATE_ZMQ_TOPIC))
                latest_state = msg
                if not state_keys_printed:
                    print("[Monitor] g1_debug keys:", sorted(latest_state.keys()), "\n")
                    state_keys_printed = True
            except zmq.Again:
                pass

            now = time.time()
            do_print = (now - last_print) >= print_period
            do_log = logf is not None and log_period > 0 and (now - last_log) >= log_period
            if not (do_print or do_log):
                time.sleep(0.002)
                continue

            if latest_planner is None or latest_state is None:
                if do_print:
                    last_print = now
                    print(f"\r[Monitor] waiting: planner={latest_planner is not None} "
                          f"state={latest_state is not None}   ", end="", flush=True)
                continue

            rec = {"t": now}
            p = latest_planner
            rec["cmd"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in p.items()}

            # Actual FK
            actual = None
            if rm is not None:
                bq = latest_state.get("body_q")
                lhq = latest_state.get("left_hand_q")
                rhq = latest_state.get("right_hand_q")
                if bq is not None:
                    actual = fk_pelvis_relative(
                        rm, np.asarray(bq),
                        np.asarray(lhq) if lhq is not None else None,
                        np.asarray(rhq) if rhq is not None else None,
                    )

            roll, pitch, base_yaw = (None, None, None)
            if "base_quat" in latest_state:
                roll, pitch, base_yaw = base_euler_deg(latest_state["base_quat"])
            rec["base_roll_deg"], rec["base_pitch_deg"], rec["base_yaw_deg"] = roll, pitch, base_yaw
            # delta_heading: robot's accumulated commanded heading (radians) from the C++ deploy.
            dh = latest_state.get("delta_heading")
            rec["delta_heading"] = float(np.asarray(dh).flat[0]) if dh is not None else None
            cmd_face_yaw = facing_yaw_deg(p.get("facing") or [1, 0, 0])
            rec["cmd_facing_yaw_deg"] = cmd_face_yaw
            # Base translation (world frame) — the robot's actual walked trajectory.
            for k in ("base_trans_measured", "base_trans_target"):
                v = latest_state.get(k)
                if v is not None:
                    rec[k] = np.asarray(v, dtype=np.float64).flatten().tolist()

            # Always compute actual/err into rec (so the file has it); print conditionally.
            lines = []
            tilt = f"tilt r/p={roll:+.1f}/{pitch:+.1f}°" if roll is not None else "tilt n/a"
            head = (f"cmdFaceYaw={cmd_face_yaw:+.0f}° baseYaw={base_yaw:+.0f}° "
                    f"dHead={rec['delta_heading']:+.2f}" if base_yaw is not None else "yaw n/a")
            lines.append(
                f"mode={p.get('mode')} move={fmt3(p.get('movement') or [0,0,0])} "
                f"face={fmt3(p.get('facing') or [0,0,0])} speed={p.get('speed')} "
                f"height={p.get('height')}  {tilt}  {head}"
            )
            if "vr_pos" in p:
                cmd_pts = np.asarray(p["vr_pos"]).reshape(3, 3)
                for i, lbl in enumerate(POINT_LABELS):
                    cmd = cmd_pts[i]
                    if actual is not None:
                        act = actual[VR_TARGET_FRAMES[i]][0]
                        err = np.linalg.norm(cmd - act)
                        lines.append(f"  {lbl:5s} cmd{fmt3(cmd)}  act{fmt3(act)}  |err|={err:.3f}m")
                        rec.setdefault("actual", {})[VR_TARGET_FRAMES[i]] = act.tolist()
                        rec.setdefault("err_m", {})[lbl] = float(err)
                    else:
                        lines.append(f"  {lbl:5s} cmd{fmt3(cmd)}  act n/a")

            if do_log:
                last_log = now
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

            if do_print:
                last_print = now
                print("\n".join(lines))
                print("-" * 78)

    except KeyboardInterrupt:
        print("\n[Monitor] stopped.")
    finally:
        if logf is not None:
            logf.close()
        planner_sub.close()
        state_sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
