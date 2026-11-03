"""Run a pi0.5 HOMIE-action policy (openpi policy server) on the real G1 through run_g1_control_loop.py.

The observation is built exactly as record_teleop_homie.py records it (state 39: 29 body joints in ScaleBFM order,
Dex1 grippers in sim finger units, head joints, base velocity; the time-aligned ego_view / left_wrist / right_wrist
camera set), the images are resized to 224x224 with padding before sending (as openpi's own clients do; the server
would do the same). The policy's 28-D rows (30 fps: 7 HOMIE commands, 17 waist + arm targets, 2 grippers, 2 head) are
published at 30 Hz as the control loop's goal (as the PICO teleop loop does) and as head/target.

Asynchronous, as real time requires: the control thread keeps executing the current chunk while the next request is
in flight; when it returns, the new chunk takes over at the row the robot has reached. With --rtc the request carries
the old chunk's remaining rows and the expected delay (real-time chunking: the server guides the new chunk onto them).

Safety: commands clipped (walk speed, height <= 0.74 like the PICO teleop, torso), arm targets clipped to the training
range (--norm-stats, q01 / q99 +- --joint-margin) and rate limited (--max-joint-speed); Ctrl+C (or the end) holds the
current pose with zero walk. Stop the PICO teleop loop first: both publish the control goal.

    python decoupled_wbc/scripts/run_policy_homie.py --policy-host lamb.mech.northwestern.edu \\
        --norm-stats /path/to/30000/assets/kjswaroopNU/g1_cube_right_homie/norm_stats.json --dry-run
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy
from sensor_msgs.msg import JointState
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_teleop_homie as rec  # noqa: E402

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC  # noqa: E402
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model  # noqa: E402
from decoupled_wbc.control.utils.ros_utils import ROSMsgPublisher  # noqa: E402

# head_control_node joint angles (MagicSim convention, rad), as run_teleop_policy_loop.py --pico-head-tracking
HEAD_YAW_LIMITS = (np.radians(-12.8), np.radians(77.2))
HEAD_PITCH_LIMITS = (0.0, np.radians(72.0))
CAMERA_KEYS = {"ego_view": "observation/head", "left_wrist": "observation/left_wrist",
               "right_wrist": "observation/right_wrist"}


@dataclass
class Args:
    policy_host: str = "lamb.mech.northwestern.edu"
    policy_port: int = 8000
    prompt: str = "Pick up the rubiks cube."
    norm_stats: Optional[Path] = None
    """The checkpoint's norm_stats.json: arm targets are clipped to the training q01..q99 +- joint_margin."""
    joint_margin: float = 0.1
    max_joint_speed: float = 1.5
    """rad/s: per-row change of every arm target is limited to this (a jump between chunks cannot whip the arm)."""
    fps: int = 30
    """Policy row rate (the dataset's)."""
    execute: int = 6
    """Rows executed per query (6 = 0.2 s at 30 fps, as in the sim tests)."""
    rtc: bool = True
    max_seconds: float = 60.0
    max_vx: float = 0.2
    max_vy: float = 0.15
    max_wz: float = 0.3
    min_height: float = 0.62
    max_height: float = 0.74
    max_torso: float = 0.2
    gripper_open: float = 5.5
    grip_debounce: int = 10
    """Rows (at fps) the policy must ask for the other gripper state before the gripper switches (0 = off)."""
    """Dex1 angle when open (0 = closed); the policy's gripper command > 0.5 closes."""
    interface: str = "enp0s31f6"
    camera_url: str = "tcp://192.168.123.164:5555"
    dry_run: bool = False
    """Query the policy and print what would be sent, publish nothing."""
    out: Path = Path("/tmp/run_policy_homie.npz")


class Runner:
    def __init__(self, args: Args):
        self.args = args
        self.obs = rec.Recorder(rec.Args(out=Path("/tmp/unused"), interface=args.interface,
                                         camera_url=args.camera_url))
        model = instantiate_g1_robot_model(waist_location="upper_body")
        upper_names = [model.joint_names[i] for i in model.get_joint_group_indices("upper_body")]
        self.base_upper = model.get_initial_upper_body_pose().astype(np.float64)
        # (upper_body slot, policy action column) for the 17 waist + arm joints
        self.slots = np.array([upper_names.index(n) for n in rec.UPPER_JOINTS])
        self.upper_in_state = np.array([rec.SCALEBFM_JOINTS.index(n) for n in rec.UPPER_JOINTS])
        self.cols = np.arange(7, 24)
        self.lo = np.full(17, -np.inf)
        self.hi = np.full(17, np.inf)
        if args.norm_stats is not None:
            stats = json.loads(args.norm_stats.read_text())["norm_stats"]["actions"]
            self.lo = np.asarray(stats["q01"])[7:24] - args.joint_margin
            self.hi = np.asarray(stats["q99"])[7:24] + args.joint_margin
        self.policy = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
        print(f"policy server: {self.policy.get_server_metadata()}")
        if not args.dry_run:
            self.goal_pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
            self.head_pub = self.obs.node.create_publisher(JointState, "head/target", 10)
        self.lock = threading.Lock()
        self.chunk, self.row, self.executed = None, 0, 0  # current chunk, next row, rows executed in total
        self.last_upper = None
        self.grip_closed, self.grip_count = [False, False], [0, 0]  # debounced gripper state, rows asking to switch
        self.stop = False
        self.log = {"t": [], "state": [], "chunk": [], "infer_ms": [], "start_row": []}

    # ------------------------------------------------------------------ observation
    def observe(self):
        """(state 39, request images) of the latest inputs; None until everything has arrived once."""
        o = self.obs
        o.poll()
        if o.state is None or o.head is None or o.images is None:
            return None
        s = o.state
        q = np.asarray(s["q"], np.float64)
        with o.lock:
            head = o.head.copy()
        gyro = np.asarray(s["floating_base_vel"], np.float64)[3:6]
        state = np.concatenate([q[o.body_idx], [rec.dex1_to_sim(o.dex1_q["left"]), rec.dex1_to_sim(o.dex1_q["right"])],
                                head, np.zeros(3), gyro]).astype(np.float32)
        images = {}
        for name, key in CAMERA_KEYS.items():
            bgr = cv2.imdecode(np.frombuffer(o.images[name], np.uint8), cv2.IMREAD_COLOR)
            images[key] = image_tools.resize_with_pad(bgr[..., ::-1], 224, 224)
        return state, images

    # ------------------------------------------------------------------ output
    def send(self, row: np.ndarray, hold: bool = False):
        a = self.args
        nav = np.zeros(3) if hold else np.clip(row[0:3], [-a.max_vx, -a.max_vy, -a.max_wz], [a.max_vx, a.max_vy, a.max_wz])
        height = float(np.clip(row[3], a.min_height, a.max_height))
        torso = np.clip(row[4:7], -a.max_torso, a.max_torso)
        target = np.clip(row[self.cols], self.lo, self.hi)
        if self.last_upper is not None:  # rate limit
            step = a.max_joint_speed / a.fps
            target = self.last_upper + np.clip(target - self.last_upper, -step, step)
        self.last_upper = target
        upper = self.base_upper.copy()
        upper[self.slots] = target
        for i, g in enumerate(row[24:26]):  # debounce: the policy's output flips between chunks
            if (g > 0.5) != self.grip_closed[i] and not hold:
                self.grip_count[i] += 1
                if self.grip_count[i] >= a.grip_debounce:
                    self.grip_closed[i], self.grip_count[i] = not self.grip_closed[i], 0
            else:
                self.grip_count[i] = 0
        grip = [0.0 if c else a.gripper_open for c in self.grip_closed]
        head = [float(np.clip(row[26], *HEAD_YAW_LIMITS)), float(np.clip(row[27], *HEAD_PITCH_LIMITS))]
        if a.dry_run:
            return
        now = time.monotonic()
        self.goal_pub.publish({"target_upper_body_pose": upper, "target_time": now + 1.0 / a.fps,
                               "navigate_cmd": nav, "base_height_command": height, "torso_orientation_rpy": torso,
                               "left_gripper_cmd": grip[0], "right_gripper_cmd": grip[1], "timestamp": now})
        msg = JointState()
        msg.header.stamp = self.obs.node.get_clock().now().to_msg()
        msg.name = ["joint_yaw", "joint_pitch"]
        msg.position = head
        self.head_pub.publish(msg)

    def control_loop(self):
        """30 Hz: execute the current chunk row by row; past its end, hold the last row with zero walk."""
        period = 1.0 / self.args.fps
        tick = time.monotonic()
        while not self.stop:
            with self.lock:
                if self.chunk is not None:
                    hold = self.row >= len(self.chunk)
                    row = self.chunk[min(self.row, len(self.chunk) - 1)]
                    self.row += 1
                    self.executed += 0 if hold else 1
                else:
                    row, hold = None, True
            if row is not None:
                self.send(row, hold=hold)
            tick += period
            time.sleep(max(0.0, tick - time.monotonic()))

    # ------------------------------------------------------------------ main
    def run(self):
        a = self.args
        print("waiting for state, head, Dex1 and cameras ...", flush=True)
        while self.observe() is None:
            time.sleep(0.05)
        # warm-up: a compiled server (serve_policy_rtc.py --compile default) compiles on its first plain and first RTC
        # request (~1-2 min); do both now, with the real observation, so no query in the run waits for it
        state, images = self.observe()
        request = {"observation/state": state, **images, "prompt": a.prompt}
        for i in range(3):
            t0 = time.monotonic()
            chunk = np.asarray(self.policy.infer(request)["actions"], np.float64)
            print(f"warm-up {i + 1}/3: {1000 * (time.monotonic() - t0):.0f} ms", flush=True)
            if a.rtc:
                request = {**request, "rtc_prev_actions": chunk[a.execute:].astype(np.float32),
                           "rtc_inference_delay": 9, "rtc_prefix_attention_horizon": len(chunk) - a.execute,
                           "rtc_use_vjp": False}
        print(f"ready. Robot standing, PICO teleop stopped, gantry on? Enter starts "
              f"({'DRY RUN, nothing is published' if a.dry_run else 'the policy drives the robot'}), Ctrl+C stops.")
        input()
        state, _ = self.observe()
        self.last_upper = state[self.upper_in_state].astype(np.float64)  # the rate limit starts from the measured arms
        if not a.dry_run:
            # the episodes start with open grippers; without a gripper command the control loop closes them (0.0)
            hold = np.zeros(28)
            hold[3], hold[self.cols], hold[26:28] = a.max_height, self.last_upper, state[31:33]
            print("opening the grippers (2 s) ...", flush=True)
            for _ in range(2 * a.fps):
                self.send(hold, hold=True)
                time.sleep(1.0 / a.fps)
            state, _ = self.observe()
            print(f"gripper state (sim units, open -0.020): {np.round(state[29:31], 3).tolist()}", flush=True)
        control = threading.Thread(target=self.control_loop, daemon=True)
        control.start()
        start, delay_rows = time.monotonic(), 0
        try:
            while time.monotonic() - start < a.max_seconds:
                # query once the current chunk has `execute` rows done (the first query at once)
                while self.chunk is not None and self.row < a.execute and not self.stop:
                    time.sleep(0.002)
                state, images = self.observe()
                request = {"observation/state": state, **images, "prompt": a.prompt}
                with self.lock:
                    at_query = self.executed
                    left = None if self.chunk is None else self.chunk[min(self.row, len(self.chunk)):]
                if a.rtc and left is not None and len(left) > 0:
                    request.update({"rtc_prev_actions": left.astype(np.float32),
                                    "rtc_inference_delay": int(min(delay_rows, len(left))),
                                    "rtc_prefix_attention_horizon": len(left), "rtc_use_vjp": False})
                t0 = time.monotonic()
                chunk = np.asarray(self.policy.infer(request)["actions"], np.float64)
                infer_ms = 1000 * (time.monotonic() - t0)
                with self.lock:
                    done = self.executed - at_query  # rows the robot executed while the request was in flight
                    self.chunk, self.row = chunk, min(done, len(chunk) - 1)
                delay_rows = int(np.ceil(infer_ms / 1000 * a.fps))
                for k, v in (("t", t0 - start), ("state", state), ("chunk", chunk), ("infer_ms", infer_ms),
                             ("start_row", done)):
                    self.log[k].append(v)
                r = chunk[min(done, len(chunk) - 1)]
                print(f"t {t0 - start:5.1f} s  infer {infer_ms:4.0f} ms  start row {done:2d}  cmd vx,vy,wz,h "
                      f"{np.round(r[:4], 2).tolist()}  grip R {r[25]:.2f}  head {np.round(r[26:28], 2).tolist()}",
                      flush=True)
        except KeyboardInterrupt:
            print("\nCtrl+C: holding the current pose")
        finally:
            with self.lock:  # hold: the last executed row, zero walk
                if self.chunk is not None:
                    self.chunk = self.chunk[max(self.row - 1, 0):max(self.row, 1)]
                    self.row = len(self.chunk)
            np.savez(a.out, **{k: np.asarray(v) for k, v in self.log.items()})
            print(f"log: {a.out}. Holding the pose (zero walk); Ctrl+C again exits (the control loop then holds on its own)")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            self.stop = True


if __name__ == "__main__":
    Runner(tyro.cli(Args)).run()
