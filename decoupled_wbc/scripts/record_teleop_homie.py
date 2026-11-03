"""Record real-robot teleop episodes in MagicSim's HOMIE layout (the sim dataset `--action homie --fps 30`).

Runs in the decoupled_wbc container beside run_g1_control_loop.py (--waist-location upper_body) and the PICO teleop
loop, with the robot's head node, Dex1 gripper server and head_zmq_bridge cameras up (launch_g1.sh). One row per
time-aligned camera set (the cameras run at 30 fps), with the latest state and teleop command:

  state (39):  29 body joints (ScaleBFM order), grippers [left, right] (sim finger position, m), head [yaw, pitch]
               (sim joint angles, rad), base linear velocity [0, 0, 0] (not measured on the robot; the policy drops
               it), base angular velocity (pelvis IMU gyro)
  action (28): HOMIE command [vx, vy, wz, base_height, torso_roll, torso_pitch, torso_yaw] (the teleop goal),
               17 waist + arm targets (the teleop IK, UPPER_JOINTS order), grippers [left, right] (0 open / 1 close),
               head [yaw, pitch] (the next row's head joints, as the sim converter does)
  images:      ego_view / left_wrist / right_wrist JPEGs of the time-aligned camera set (640x480)

Episodes: PICO A starts an episode and A again saves it, B discards it (the streamer's toggle_data_collection /
toggle_data_abort in the teleop goal); keyboard on /keyboard_input: "c" start / save, "x" discard. Each saved episode
is a folder <out>/episode_<n>/ with data.npz (state, action, timestamps, prompt) and <camera>/<frame>.jpg;
MagicSim's convert_real_to_lerobot.py turns the folder into a LeRobot v2.1 dataset next to the sim one.

    python decoupled_wbc/scripts/record_teleop_homie.py --out /home/kjs/wbcG1/real_data/cube_right \\
        --prompt "Pick up the rubiks cube." --interface enp0s31f6
"""

import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import threading
import time

import msgpack
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import String as RosStringMsg
import tyro
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
import zmq

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC, KEYBOARD_INPUT_TOPIC, STATE_TOPIC_NAME
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgSubscriber

FPS = 30  # = the camera rate: one row per camera set
# MagicSim dataset layout (convert_raw_to_lerobot_g1_dex1_head.py)
SCALEBFM_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
UPPER_JOINTS = [n for n in SCALEBFM_JOINTS if n.startswith(("waist", "left_shoulder", "right_shoulder", "left_elbow",
                                                            "right_elbow", "left_wrist", "right_wrist"))]  # fmt: skip
CAMERAS = ("ego_view", "left_wrist", "right_wrist")
# Dex1: real motor angle 5.5 rad = open, 0 = closed; sim finger joint -0.02 m = open, 0.0245 m = closed
# (robot_server DEX1_OPEN / DEX1_CLOSE). Linear between the two endpoints (not calibrated against the fingers yet).
DEX1_OPEN_RAD, SIM_OPEN, SIM_CLOSE = 5.5, -0.02, 0.0245
DEFAULT_HEIGHT = 0.74


def dex1_to_sim(q_rad: float) -> float:
    closed = 1.0 - min(max(q_rad / DEX1_OPEN_RAD, 0.0), 1.0)
    return SIM_OPEN + closed * (SIM_CLOSE - SIM_OPEN)


@dataclass
class Args:
    out: Path
    """Folder for the raw episodes (episode_<n>/ is added per saved episode)."""
    prompt: str = "Pick up the rubiks cube."
    interface: str = "enp0s31f6"
    """Laptop network interface on the robot's 192.168.123.x network (Unitree DDS: Dex1 state)."""
    camera_url: str = "tcp://192.168.123.164:5555"
    max_seconds: float = 60.0
    """An episode longer than this is discarded (forgotten stop)."""


class Recorder:
    def __init__(self, args: Args):
        self.args = args
        self.lock = threading.Lock()
        robot_model = instantiate_g1_robot_model(waist_location="upper_body")
        names = list(robot_model.joint_names)
        self.body_idx = np.array([names.index(n) for n in SCALEBFM_JOINTS])
        upper_names = [names[i] for i in robot_model.get_joint_group_indices("upper_body")]
        missing = [n for n in UPPER_JOINTS if n not in upper_names]
        if missing:
            raise RuntimeError(f"upper_body group lacks {missing}: run the teleop loop with waist_location=upper_body")
        self.upper_idx = np.array([upper_names.index(n) for n in UPPER_JOINTS])

        ros = ROSManager(node_name="TeleopRecorder")
        self.node = ros.node
        self.goal_sub = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)
        self.state_sub = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.node.create_subscription(JointState, "head/state", self.on_head, 10)
        self.node.create_subscription(RosStringMsg, KEYBOARD_INPUT_TOPIC, self.on_key, 10)
        # (ROSManager already spins this node in its own thread)

        ChannelFactoryInitialize(0, args.interface)
        self.dex1 = {}
        for side in ("left", "right"):
            sub = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
            sub.Init(None, 1)  # read on demand (a handler would decode 500 msgs/s in Python)
            self.dex1[side] = sub
        self.dex1_q = {"left": DEX1_OPEN_RAD, "right": DEX1_OPEN_RAD}

        self.cam = zmq.Context.instance().socket(zmq.SUB)
        self.cam.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cam.setsockopt(zmq.CONFLATE, 1)
        self.cam.connect(args.camera_url)

        self.goal, self.state, self.head, self.images = None, None, None, None
        self.new_images = False  # a camera set arrived since the last row
        self.toggles = []  # "start_stop" / "abort" events from the PICO or the keyboard

    # ------------------------------------------------------------------ inputs
    def on_head(self, msg):
        angles = dict(zip(msg.name, msg.position))
        with self.lock:
            self.head = np.array([angles["joint_yaw"], angles["joint_pitch"]], np.float64)

    def on_key(self, msg):
        if msg.data == "c":
            self.toggles.append("start_stop")
        elif msg.data == "x":
            self.toggles.append("abort")

    def poll(self):
        """Latest goal / state / Dex1 / cameras (each keeps its last value when nothing new arrived)."""
        goal = self.goal_sub.get_msg()
        if goal is not None:
            if goal.get("toggle_data_collection"):
                self.toggles.append("start_stop")
            if goal.get("toggle_data_abort"):
                self.toggles.append("abort")
            self.goal = goal
        state = self.state_sub.get_msg()
        if state is not None:
            self.state = state
        # the SDK prints "[Reader] take sample error" whenever no new sample came within the timeout (normal at
        # 500 Hz: the last value is kept), so its output is swallowed here
        with contextlib.redirect_stdout(io.StringIO()):
            msgs = {side: sub.Read(timeout=0.001) for side, sub in self.dex1.items()}
        for side, msg in msgs.items():
            if msg is not None:
                self.dex1_q[side] = float(msg.states[0].q)
        try:
            self.images = msgpack.unpackb(self.cam.recv(zmq.NOBLOCK), raw=False)["images"]
            self.new_images = True
        except zmq.Again:
            pass

    def ready(self):
        missing = [n for n, v in (("teleop goal", self.goal), ("control loop state", self.state), ("head/state", self.head),
                                  ("cameras", self.images)) if v is None]
        return missing

    def frame(self):
        """(state 39, action 28 without the head, jpgs) of the latest inputs."""
        g, s = self.goal, self.state
        q = np.asarray(s["q"], np.float64)
        with self.lock:
            head = self.head.copy()
        gyro = np.asarray(s["floating_base_vel"], np.float64)[3:6]
        state = np.concatenate([q[self.body_idx], [dex1_to_sim(self.dex1_q["left"]), dex1_to_sim(self.dex1_q["right"])],
                                head, np.zeros(3), gyro])
        nav = np.asarray(g.get("navigate_cmd", np.zeros(3)), np.float64)
        rpy = np.asarray(g.get("torso_orientation_rpy", np.zeros(3)), np.float64)
        height = float(g.get("base_height_command", DEFAULT_HEIGHT))
        upper = np.asarray(g["target_upper_body_pose"], np.float64)[self.upper_idx]
        # gripper command: the teleop sends the Dex1 angle (5.5 open); closing below half
        grips = [float(g.get(f"{side}_gripper_cmd", DEX1_OPEN_RAD) < DEX1_OPEN_RAD / 2) for side in ("left", "right")]
        action = np.concatenate([nav, [height], rpy, upper, grips])
        return state, action, {name: self.images[name] for name in CAMERAS}

    # ------------------------------------------------------------------ episodes
    def run(self):
        self.args.out.mkdir(parents=True, exist_ok=True)
        n_saved = len(list(self.args.out.glob("episode_*")))
        print(f"recording to {self.args.out} ({n_saved} episodes there); PICO A / key 'c' = start or save, "
              f"B / 'x' = discard", flush=True)
        recording, frames, last_report = False, [], 0.0
        while rclpy.ok():
            self.poll()
            while self.toggles:
                event = self.toggles.pop(0)
                if event == "start_stop" and not recording:
                    missing = self.ready()
                    if missing:
                        print(f"cannot start, no data yet from: {', '.join(missing)}", flush=True)
                        continue
                    recording, frames = True, []
                    print(f"\n>>> episode {n_saved}: recording ('{self.args.prompt}')", flush=True)
                elif event == "start_stop" and recording:
                    recording = False
                    if len(frames) < FPS:
                        print("episode shorter than 1 s: discarded", flush=True)
                    else:
                        self.save(n_saved, frames)
                        n_saved += 1
                elif event == "abort" and recording:
                    recording = False
                    print(f"episode discarded ({len(frames) / FPS:.1f} s)", flush=True)
            if recording and self.new_images:  # one row per camera set
                self.new_images = False
                frames.append((time.monotonic(), *self.frame()))
                if len(frames) > self.args.max_seconds * FPS:
                    recording = False
                    print(f"episode longer than {self.args.max_seconds} s: discarded", flush=True)
                elif time.monotonic() - last_report > 1.0:
                    last_report = time.monotonic()
                    print(f"  {len(frames) / FPS:5.1f} s", end="\r", flush=True)
            time.sleep(0.002)

    def save(self, index: int, frames: list):
        ep = self.args.out / f"episode_{index:06d}"
        ep.mkdir(parents=True)
        stamps = np.array([f[0] for f in frames])
        state = np.stack([f[1] for f in frames]).astype(np.float32)
        action = np.zeros((len(frames), 28), np.float32)
        action[:, :26] = np.stack([f[2] for f in frames])
        action[:, 26:28] = np.concatenate([state[1:, 31:33], state[-1:, 31:33]])  # head: the next frame's joints
        for cam in CAMERAS:
            (ep / cam).mkdir()
            for i, f in enumerate(frames):
                (ep / cam / f"{i:06d}.jpg").write_bytes(f[3][cam])
        np.savez(ep / "data.npz", state=state, action=action, timestamps=stamps)
        (ep / "meta.json").write_text(json.dumps({"prompt": self.args.prompt, "fps": FPS, "frames": len(frames),
                                                  "seconds": round(len(frames) / FPS, 2)}))
        dt = np.diff(stamps)
        print(f"saved {ep.name}: {len(frames) / FPS:.1f} s, frame period mean {1000 * dt.mean():.1f} ms "
              f"(max {1000 * dt.max():.0f} ms)", flush=True)


def main(args: Args):
    Recorder(args).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
