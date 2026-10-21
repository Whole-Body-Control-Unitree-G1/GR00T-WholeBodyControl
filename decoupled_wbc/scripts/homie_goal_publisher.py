"""Drive run_g1_control_loop.py with MagicSim HOMIE-action episodes (sim-to-sim / open-loop real replay).

Publishes the control loop's goal (CONTROL_GOAL_TOPIC) at 50 Hz from a LeRobot episode converted with
``convert_raw_to_lerobot_g1_dex1_head.py --action homie`` (28-D: 7 HOMIE commands, 17 waist + arm joints in MagicSim's
UPPER_JOINTS order, 2 grippers, 2 head), and logs the robot state for comparison with the recorded pelvis.

Run the control loop with the MagicSim HOMIE settings and the ROS keyboard dispatcher, e.g. in MuJoCo:

  python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface sim --waist_location upper_body \\
      --stand_cmd_threshold 0.1 --keyboard_dispatcher_type ros
  python decoupled_wbc/scripts/homie_goal_publisher.py --episode episode_000006.parquet --info info.json --release_band

Sequence: hold the episode's first upper-body pose and commands (ramped in over --ramp s), activate the lower-body
policy (key "]"), release the MuJoCo elastic band (key "9", --release_band, sim only), settle, replay every frame,
then hold the last pose with zero velocity. Commands are clipped to the --max_* limits.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from std_msgs.msg import String as RosStringMsg
import tyro

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    KEYBOARD_INPUT_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber

FPS = 50
CMD_NAMES = ["cmd_vx", "cmd_vy", "cmd_wz", "cmd_base_height", "cmd_torso_roll", "cmd_torso_pitch", "cmd_torso_yaw"]


@dataclass
class Args:
    episode: Path
    """LeRobot parquet episode (HOMIE action layout)."""
    info: Path
    """The dataset's meta/info.json (action names)."""
    out: Path = Path("homie_replay_log.npz")
    """Robot state log: time, floating base pose, joint positions, commands sent."""
    release_band: bool = False
    """Send "9" after activation: releases the MuJoCo elastic band (sim only)."""
    ramp: float = 2.0
    """Seconds to move the upper body to the episode's first pose."""
    settle: float = 3.0
    """Seconds to stand after activation (and band release) before replaying."""
    hold_end: float = 3.0
    """Seconds to hold the last pose with zero velocity after the episode."""
    gripper_close: float = 5.5
    """Real Dex1 gripper angle for the binary "close" command (0 = open)."""
    max_vx: float = 0.3
    max_vy: float = 0.2
    max_wz: float = 0.4
    min_height: float = 0.65
    max_height: float = 0.75
    max_torso: float = 0.2


def load_episode(args: Args):
    names = json.loads(args.info.read_text())["features"]["action"]["names"]
    action = np.stack(pd.read_parquet(args.episode)["action"].to_numpy()).astype(np.float64)
    pelvis = np.stack(pd.read_parquet(args.episode)["observation.pelvis_pose"].to_numpy())
    col = {n: i for i, n in enumerate(names)}
    cmd = action[:, [col[n] for n in CMD_NAMES]]
    cmd[:, 0] = cmd[:, 0].clip(-args.max_vx, args.max_vx)
    cmd[:, 1] = cmd[:, 1].clip(-args.max_vy, args.max_vy)
    cmd[:, 2] = cmd[:, 2].clip(-args.max_wz, args.max_wz)
    cmd[:, 3] = cmd[:, 3].clip(args.min_height, args.max_height)
    cmd[:, 4:7] = cmd[:, 4:7].clip(-args.max_torso, args.max_torso)
    grippers = action[:, [col["left_gripper"], col["right_gripper"]]] * args.gripper_close
    return names, col, action, cmd, grippers, pelvis


def main(args: Args):
    names, col, action, cmd, grippers, recorded_pelvis = load_episode(args)
    robot_model = instantiate_g1_robot_model(waist_location="upper_body")
    upper_idx = robot_model.get_joint_group_indices("upper_body")
    upper_names = [robot_model.joint_names[i] for i in upper_idx]
    base_pose = robot_model.get_initial_upper_body_pose().astype(np.float64)
    mapped = [(k, col[n]) for k, n in enumerate(upper_names) if n in col]  # (upper_body slot, action column)
    missing = [n for n in upper_names if n not in col]
    print(f"upper_body group: {len(upper_names)} joints, {len(mapped)} from the episode; held at the initial pose: "
          f"{missing}")

    def upper_pose(t):
        pose = base_pose.copy()
        for slot, column in mapped:
            pose[slot] = action[t, column]
        return pose

    ros = ROSManager(node_name="HomieGoalPublisher")
    goal_pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    key_pub = ros.node.create_publisher(RosStringMsg, KEYBOARD_INPUT_TOPIC, 10)
    state_sub = ROSMsgSubscriber(STATE_TOPIC_NAME)

    def key(k):
        msg = RosStringMsg()
        msg.data = k
        key_pub.publish(msg)
        print(f"key {k!r}")

    log = {"t": [], "phase": [], "base_pose": [], "q": [], "cmd": []}

    def record(phase, sent_cmd):
        state = state_sub.get_msg()
        if state is None:
            return
        log["t"].append(time.monotonic())
        log["phase"].append(phase)
        log["base_pose"].append(np.asarray(state["floating_base_pose"], dtype=np.float64))
        log["q"].append(np.asarray(state["q"], dtype=np.float64))
        log["cmd"].append(np.asarray(sent_cmd, dtype=np.float64))

    def publish(t, nav, phase, target_dt=1.0 / FPS):
        now = time.monotonic()
        goal_pub.publish({
            "target_upper_body_pose": upper_pose(t),
            "target_time": now + target_dt,
            "navigate_cmd": nav,
            "base_height_command": float(cmd[t, 3]),
            "torso_orientation_rpy": cmd[t, 4:7],
            "left_gripper_cmd": float(grippers[t, 0]),
            "right_gripper_cmd": float(grippers[t, 1]),
        })
        record(phase, np.concatenate([nav, cmd[t, 3:7]]))

    print("waiting for the control loop state ...")
    while state_sub.get_msg() is None:
        time.sleep(0.1)

    zero = np.zeros(3)
    publish(0, zero, 0, target_dt=args.ramp)  # ramp to the first pose
    t_end = time.monotonic() + args.ramp
    while time.monotonic() < t_end:
        time.sleep(1.0 / FPS)
        publish(0, zero, 0, target_dt=max(t_end - time.monotonic(), 1.0 / FPS))
    key("]")
    if args.release_band:
        time.sleep(1.0)
        key("9")
    t_end = time.monotonic() + args.settle
    while time.monotonic() < t_end:
        publish(0, zero, 1)
        time.sleep(1.0 / FPS)

    print(f"replaying {len(action)} frames ({len(action) / FPS:.1f} s)")
    start = time.monotonic()
    for t in range(len(action)):
        publish(t, cmd[t, :3], 2)
        time.sleep(max(0.0, start + (t + 1) / FPS - time.monotonic()))
    t_end = time.monotonic() + args.hold_end
    while time.monotonic() < t_end:
        publish(len(action) - 1, zero, 3)
        time.sleep(1.0 / FPS)

    np.savez(args.out, recorded_pelvis=recorded_pelvis, recorded_cmd=cmd, joint_names=np.array(robot_model.joint_names),
             **{k: np.asarray(v) for k, v in log.items()})
    print(f"saved {len(log['t'])} state samples to {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
