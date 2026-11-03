import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
import tyro

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from decoupled_wbc.control.main.teleop.configs.configs import TeleopConfig
from decoupled_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from decoupled_wbc.control.utils.telemetry import Telemetry

TELEOP_NODE_NAME = "TeleopPolicy"
# --pico-head-tracking: head_control_node's joint angles (MagicSim convention, rad): straight ahead and level, and the
# range in which the head camera does not see the robot's own body (the sim collection's head_limits)
HEAD_LEVEL = (0.562, 0.157)
HEAD_YAW_LIMITS = (np.radians(-12.8), np.radians(77.2))
HEAD_PITCH_LIMITS = (0.0, np.radians(72.0))
GRIPPER_OPEN = 5.5  # Dex1 command (rad): open, as PicoStreamer's MAX_GRIPPER_ANGLE


def main(config: TeleopConfig):
    ros_manager = ROSManager(node_name=TELEOP_NODE_NAME)
    node = ros_manager.node

    if config.robot == "g1":
        # as run_g1_control_loop.py: "upper_body" = the teleop IK drives the waist, the lower body only the legs
        waist_location = config.waist_location or ("lower_and_upper_body" if config.enable_waist else "lower_body")
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
        )
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    if config.lerobot_replay_path:
        teleop_policy = LerobotReplayPolicy(
            robot_model=robot_model, parquet_path=config.lerobot_replay_path
        )
    else:
        print("running teleop policy, waiting teleop policy to be initialized...")
        retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=config.enable_visualization,
            body_active_joint_groups=["upper_body"],
        )
        teleop_policy = TeleopPolicy(
            robot_model=robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=config.body_control_device,
            hand_control_device=config.hand_control_device,
            body_streamer_ip=config.body_streamer_ip,  # vive tracker, leap motion does not require
            body_streamer_keyword=config.body_streamer_keyword,
            enable_real_device=config.enable_real_device,
            replay_data_path=config.teleop_replay_path,
            pico_head_tracking=config.pico_head_tracking,
        )

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    head_publisher = node.create_publisher(JointState, "head/target", 10) if config.pico_head_tracking else None

    def publish_head(yaw, pitch):
        head = JointState()
        head.header.stamp = node.get_clock().now().to_msg()
        head.name = ["joint_yaw", "joint_pitch"]
        head.position = [float(yaw), float(pitch)]
        head_publisher.publish(head)

    # --pico-pause-home: the robot starts paused at the startup pose; pausing again homes the arms (see TeleopConfig)
    body_streamer = getattr(getattr(teleop_policy, "teleop_streamer", None), "body_streamer", None)
    if hasattr(body_streamer, "current_base_height"):
        body_streamer.current_base_height = config.pico_base_height
    pause_home = config.pico_pause_home and hasattr(teleop_policy, "is_active")
    was_active = False
    blend_until = 0.0  # target_time of the home move (paused) or of the blend into teleop (just activated)
    held_height = config.pico_base_height

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    telemetry = Telemetry(window_size=100)

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()

                # Add timing information to the message
                t_now = time.monotonic()
                data["timestamp"] = t_now

                # Set target completion time - longer for initial pose, then match control frequency
                if iteration == 0:
                    data["target_time"] = t_now + time_to_get_to_initial_pose
                else:
                    data["target_time"] = t_now + (1 / config.teleop_frequency)

                paused = False
                if pause_home:
                    active = teleop_policy.is_active
                    if was_active and not active:
                        # paused: back to the startup pose (the IK's default output) along one slow straight line;
                        # the interpolator re-plans from the current pose to the same target_time on every tick
                        teleop_policy.retargeting_ik.reset()
                        data["target_upper_body_pose"] = teleop_policy.retargeting_ik.get_action()
                        blend_until = t_now + config.home_seconds
                        print(f"[pause] homing the arms ({config.home_seconds:.0f} s), grippers open, head level; "
                              f"joysticks still walk", flush=True)
                    elif active and not was_active and iteration > 0:
                        # activated: blend from the home pose into the (freshly calibrated) teleop targets
                        blend_until = t_now + config.home_seconds / 3
                        print("[active] teleop on", flush=True)
                    was_active = active
                    paused = not active
                    if t_now < blend_until:
                        data["target_time"] = blend_until
                    if paused:
                        data["left_gripper_cmd"] = data["right_gripper_cmd"] = GRIPPER_OPEN
                        if hasattr(body_streamer, "current_base_height"):
                            body_streamer.current_base_height = held_height  # X / Y ignored while paused
                        data["base_height_command"] = held_height
                    else:
                        held_height = data.get("base_height_command", held_height)

                # Publish the teleop command
                with telemetry.timer("publish_teleop_command"):
                    control_publisher.publish(data)
                if head_publisher is not None and paused:
                    publish_head(*HEAD_LEVEL)
                elif head_publisher is not None and "head_rel_yaw_pitch" in data:
                    rel_yaw, rel_pitch = data["head_rel_yaw_pitch"]
                    publish_head(np.clip(HEAD_LEVEL[0] + rel_yaw, *HEAD_YAW_LIMITS),
                                 np.clip(HEAD_LEVEL[1] + rel_pitch, *HEAD_PITCH_LIMITS))

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Teleop Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")

    finally:
        print("Cleaning up...")
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
