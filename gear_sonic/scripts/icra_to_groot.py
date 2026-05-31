#!/usr/bin/env python3
"""
icra_to_groot.py  —  ICRA dataset replayer for GR00T data collection

Replaces pico_manager + camera bridge during data collection.
Reads the ICRA UMI dataset and replays each episode as if it were a live
PICO teleoperation session in PLANNER_VR_3PT mode (stream_mode=5).

The existing data collection pipeline is used unchanged:
  - C++ deploy  (./deploy.sh --input-type zmq_manager sim)   reads planner ZMQ → runs encoder → publishes robot state
  - run_data_exporter.py                                       records everything to LeRobot v2.1 parquet

This script provides:
  Port 5555  ZMQ PUB  —  ego_view camera frames  (msgpack, same format as camera bridge)
  Port 5556  ZMQ PUB  —  planner + manager_state  (same format as pico_manager PLANNER_VR_3PT)

Run this alongside the normal data collection pipeline:
  Terminal 1:  ./gear_sonic_deploy/deploy.sh --input-type zmq_manager sim
  Terminal 2:  python gear_sonic/scripts/run_data_exporter.py \\
                   --task-prompt "manipulation task" --with-dex1-grippers \\
                   --camera-host localhost
  Terminal 3:  python gear_sonic/scripts/icra_to_groot.py \\
                   --icra-root ~/icra-dataset \\
                   [--episodes 0-9] [--fps 30]

Coordinate conventions
----------------------
ICRA gripper poses:  camera frame (Z-forward, X-right, Y-down), scalar-last quat [qx,qy,qz,qw]
Sonic planner input: robot frame (X-forward, Y-left, Z-up),   scalar-first quat [qw,qx,qy,qz]

Frame transform  Q:
    Robot_X =  UMI_Z
    Robot_Y = -UMI_X
    Robot_Z = -UMI_Y
"""

import argparse
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pandas as pd
import zmq
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message, pack_pose_message

# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

# Q: camera frame → robot frame
Q_CAM_TO_ROBOT = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

# Rotation OFFSETS per keypoint — from pico_manager_thread_server.py
# Applied as:  new_rot = original_rot * OFFSET  (post-multiply)
# Order: [L-Wrist, R-Wrist, Neck]
OFFSETS = [
    sRot.identity(),                                           # L-Wrist (empirical: start with none)
    sRot.identity(),                                           # R-Wrist (empirical: start with none)
    sRot.from_euler("xyz", [  0,   0, -90], degrees=True),   # Neck (aligns cam-Z forward → robot convention)
]

TORSO_LINK_OFFSET_Z = 0.05   # m: root → torso_link
NECK_LINK_LENGTH    = 0.35   # m: torso_link → neck

SLOW_WALK_FLOOR = 0.1        # m/s: minimum explicit SLOW_WALK speed (pico uses 0.1)

# G1 wrist FK target positions at zero pose (pelvis-relative, robot frame).
# From MuJoCo FK: wrist_yaw_link + local offset [0.18, ∓0.025, 0] applied in identity frame.
# Left:  [0.1998+0.18, 0.1487-0.025, 0.0952] = [0.3798, 0.1237, 0.0952]
# Right: [0.1998+0.18,-0.1487+0.025, 0.0952] = [0.3798,-0.1237, 0.0952]
FK_LWRIST_POS = np.array([0.3798,  0.1237, 0.0952], dtype=np.float64)
FK_RWRIST_POS = np.array([0.3798, -0.1237, 0.0952], dtype=np.float64)
# G1 wrist FK orientation at zero pose is identity (all wrist joints = 0)
FK_WRIST_ROT = sRot.identity()


def cam_pos_to_robot(pos: np.ndarray) -> np.ndarray:
    return Q_CAM_TO_ROBOT @ pos.astype(np.float64)


def cam_quat_xyzw_to_robot_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last [x,y,z,w] camera quat → scalar-first [w,x,y,z] robot quat."""
    R_cam   = sRot.from_quat(q_xyzw.astype(np.float64))
    R_robot = sRot.from_matrix(Q_CAM_TO_ROBOT @ R_cam.as_matrix())
    return R_robot.as_quat(scalar_first=True)


def umi_hand_raw(
    pos_cam: np.ndarray, q_xyzw: np.ndarray, nk_q_xyzw: np.ndarray,
    *, head_stabilize: bool, R_w2r: sRot = None, calib_neck_inv: sRot = None,
) -> tuple[np.ndarray, sRot]:
    """Map a UMI head-camera-frame hand pose to a head-relative robot-frame pose (pre-offset).

    UMI gripper poses are expressed in the head CAMERA frame, which rotates with
    the operator's head. The head pitches ~40° looking into a bin, so any constant
    camera→robot rotation folds the human's forward+down reach into mostly vertical
    motion (the "weird arm" bug).

    head_stabilize=True (correct): use the LIVE head quaternion as camera→UMI-world,
        so per-frame head rotation is removed, then a fixed yaw R_w2r = Rz(-yaw0)
        rotates UMI world (Z-up) into robot frame (gaze→+X). Head pitch/yaw no longer
        contaminate the hand.
    head_stabilize=False (legacy): constant Q_CAM_TO_ROBOT + frame-0 neck inverse only.

    Returns (pos_robot, R_robot): head-relative hand offset and gripper orientation in
    robot frame. The caller adds neck_pos and the frame-0 calibration offsets.
    """
    if head_stabilize:
        R_cw = sRot.from_quat(nk_q_xyzw.astype(np.float64))   # camera → UMI world (Z-up)
        pos = R_w2r.apply(R_cw.apply(pos_cam.astype(np.float64)))
        rot = R_w2r * R_cw * sRot.from_quat(q_xyzw.astype(np.float64))
    else:
        pos = calib_neck_inv.apply(cam_pos_to_robot(pos_cam))
        rot = calib_neck_inv * sRot.from_quat(cam_quat_xyzw_to_robot_wxyz(q_xyzw)[[1, 2, 3, 0]])
    return pos, rot


def compute_3pt_from_umi(
    lw_pos_cam: np.ndarray, lw_q_xyzw: np.ndarray,
    rw_pos_cam: np.ndarray, rw_q_xyzw: np.ndarray,
    nk_q_xyzw:  np.ndarray,
    calib: dict,
    head_stabilize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert UMI SE(3) poses → Sonic vr_3pt_position (9,) and vr_3pt_orientation (12,).

    Key insight: UMI gripper positions are HEAD-relative (head = origin).
    Sonic needs PELVIS-relative positions. Conversion:
        wrist_pelvis = neck_pos_pelvis + neck_rot_inv @ cam_to_robot(wrist_cam) - pos_offset

    Steps:
      1. Camera → Robot frame
      2. Apply per-keypoint rotation OFFSETS
      3. Apply neck calibration (zero initial neck tilt)
      4. Neck position from kinematic chain (pelvis-relative)
      5. Wrist positions: add neck_pos to convert head-relative → pelvis-relative,
         then subtract calibration offset so frame-0 maps to G1 FK rest pose.

    All outputs: pelvis-relative, robot Z-up frame, scalar-first quaternions.
    """
    calib_neck_inv = calib["calib_neck_inv"]
    R_w2r = calib["R_world_to_robot"]

    # 1-3. Camera-frame hand → head-relative robot frame.
    #   head_stabilize=True removes per-frame head rotation via the live head quat
    #   (fixes the forward-reach loss when the head is pitched down into a bin);
    #   False = legacy constant-Q + frame-0 neck inverse. See umi_hand_raw().
    lw_pos_nk, lw_rot = umi_hand_raw(
        lw_pos_cam, lw_q_xyzw, nk_q_xyzw,
        head_stabilize=head_stabilize, R_w2r=R_w2r, calib_neck_inv=calib_neck_inv)
    rw_pos_nk, rw_rot = umi_hand_raw(
        rw_pos_cam, rw_q_xyzw, nk_q_xyzw,
        head_stabilize=head_stabilize, R_w2r=R_w2r, calib_neck_inv=calib_neck_inv)

    # 3b. Torso (3rd point) orientation: held UPRIGHT (identity).
    #     The UMI dataset only provides a HEAD quaternion in the Z-up world frame.
    #     Applying Q (camera→robot) to it is invalid (see icra_to_groot_notes.md) and
    #     mis-maps head YAW into torso ROLL — the robot's waist bends to the right as
    #     the operator turns their head. There is no valid torso/waist orientation
    #     signal in the data, so we hold the torso upright. Head yaw is still used for
    #     locomotion facing in head_delta_to_planner().
    nk_rot = sRot.identity()

    # 4. Neck position in pelvis frame (kinematic chain).
    #    With nk_rot=identity, neck_z=[0,0,1] → nk_pos = [0,0,0.40] constant (stable target).
    neck_z = nk_rot.apply([0, 0, 1])
    nk_pos = np.array([0.0, 0.0, TORSO_LINK_OFFSET_Z]) + NECK_LINK_LENGTH * neck_z

    # 5. Wrist: head-relative → pelvis-relative, then subtract FK position offset
    lw_pos_pelvis = nk_pos + lw_pos_nk - calib["lw_pos_offset"]
    rw_pos_pelvis = nk_pos + rw_pos_nk - calib["rw_pos_offset"]

    # 6. Apply wrist rotation offset (pre-multiply, same as pico_manager _apply_calibration)
    #    calibrated_rot = rot_offset * corrected_rot  →  identity at frame 0, delta thereafter
    lw_rot_cal = calib["lw_rot_offset"] * lw_rot
    rw_rot_cal = calib["rw_rot_offset"] * rw_rot

    vr_3pt_pos = np.concatenate([lw_pos_pelvis, rw_pos_pelvis, nk_pos]).astype(np.float32)
    vr_3pt_orn = np.concatenate([
        lw_rot_cal.as_quat(scalar_first=True),
        rw_rot_cal.as_quat(scalar_first=True),
        nk_rot.as_quat(scalar_first=True),
    ]).astype(np.float32)

    return vr_3pt_pos, vr_3pt_orn


def calibrate_from_first_frame(
    lw_pos_cam: np.ndarray, lw_q_xyzw: np.ndarray,
    rw_pos_cam: np.ndarray, rw_q_xyzw: np.ndarray,
    nk_q_xyzw:  np.ndarray,
    head_stabilize: bool = True,
) -> dict:
    """
    Capture calibration from episode first frame.
    1. Neck orientation zeroing (calib_neck_inv).
    2. Wrist position offset: maps measured first-frame pelvis-relative wrist
       positions onto the G1's natural hanging FK positions, so all subsequent
       frames are expressed as deltas from that rest pose.
    3. Initial head yaw in UMI world frame (for delta-based locomotion facing).
    """
    nk_rot = sRot.from_quat(cam_quat_xyzw_to_robot_wxyz(nk_q_xyzw)[[1,2,3,0]]) * OFFSETS[2]
    calib_neck_inv = nk_rot.inv()

    # Initial head yaw in UMI world frame (Z-up).
    # nk_q_xyzw is the head quaternion in UMI world (scalar-last).
    # R.apply([0,0,1]) = camera-Z axis in UMI world = looking direction.
    # arctan2(y, x) gives the horizontal yaw of that looking direction.
    head_rot_0 = sRot.from_quat(nk_q_xyzw.astype(np.float64))
    looking_0 = head_rot_0.apply([0.0, 0.0, 1.0])
    initial_head_yaw_umi = float(np.arctan2(looking_0[1], looking_0[0]))

    # Fixed UMI-world (Z-up) → robot rotation: yaw so the initial gaze lands on +X.
    R_world_to_robot = sRot.from_euler("z", -initial_head_yaw_umi)

    # First-frame wrist pose in head-relative robot frame, via the SAME mapping the
    # per-frame path uses (so frame 0 lands exactly on FK rest after these offsets).
    lw_pos_nk, lw_rot_corrected = umi_hand_raw(
        lw_pos_cam, lw_q_xyzw, nk_q_xyzw,
        head_stabilize=head_stabilize, R_w2r=R_world_to_robot, calib_neck_inv=calib_neck_inv)
    rw_pos_nk, rw_rot_corrected = umi_hand_raw(
        rw_pos_cam, rw_q_xyzw, nk_q_xyzw,
        head_stabilize=head_stabilize, R_w2r=R_world_to_robot, calib_neck_inv=calib_neck_inv)

    # At calibration frame, calibrated neck = identity → neck_z = [0,0,1] always
    nk_pos_calib = np.array([0.0, 0.0, TORSO_LINK_OFFSET_Z + NECK_LINK_LENGTH])

    lw_pos_pelvis_0 = nk_pos_calib + lw_pos_nk
    rw_pos_pelvis_0 = nk_pos_calib + rw_pos_nk

    # Rotation offset: maps measured calib rotation → G1 FK rotation (identity at zero pose)
    # rot_offset = FK_rot * inv(measured) = identity * inv(measured) = inv(measured)
    # Applied PRE-multiply each frame: calibrated_rot = rot_offset * corrected_rot
    lw_rot_offset = FK_WRIST_ROT * lw_rot_corrected.inv()
    rw_rot_offset = FK_WRIST_ROT * rw_rot_corrected.inv()

    return {
        "calib_neck_inv":      calib_neck_inv,
        "R_world_to_robot":    R_world_to_robot,
        "lw_pos_offset":       lw_pos_pelvis_0 - FK_LWRIST_POS,
        "rw_pos_offset":       rw_pos_pelvis_0 - FK_RWRIST_POS,
        "lw_rot_offset":       lw_rot_offset,
        "rw_rot_offset":       rw_rot_offset,
        "initial_head_yaw_umi": initial_head_yaw_umi,
    }


# ---------------------------------------------------------------------------
# Lower body → planner locomotion signal from head world trajectory
# ---------------------------------------------------------------------------

def head_delta_to_planner(
    prev_head_pos: np.ndarray,    # (3,) UMI world frame (Z-up)
    curr_head_pos: np.ndarray,    # (3,) UMI world frame (Z-up)
    curr_head_q_xyzw: np.ndarray, # (4,) UMI world frame, scalar-last
    dt: float,
    calib: dict,
    speed_ref: float = 0.3,       # m/s of human motion that maps to full movement magnitude (1.0)
    facing_alpha: float = 0.2,    # EMA weight for facing low-pass (1.0 = no smoothing)
    speed_deadzone: float = 0.06, # m/s (smoothed) below which we idle — set the walk duty cycle
    match_speed: bool = False,    # SLOW_WALK style: command explicit speed ≈ human speed (fixes walked distance)
    speed_max: float = 0.5,       # upper end of the SLOW_WALK ramp (m/s); pico uses 0.6
    speed_alpha: float = 0.4,     # EMA weight for the matched-speed low-pass
    walk_direction: bool = False, # steer movement along human's actual walk direction, not gaze
) -> tuple[list, list, float, float]:
    """
    Derive planner locomotion command from head world-frame delta.

    UMI world is Z-up with an unknown Z-rotation relative to robot world — Q_CAM_TO_ROBOT
    cannot map head orientations to robot frame correctly.  Instead we use delta-yaw:
      - At calibration frame (frame 0), robot faces +X (forward).
      - Each subsequent frame, we compute how much the head has yawed in UMI world
        relative to frame 0, and rotate the robot facing by the same angle.
      - Speed is derived from |delta_pos[:2]| / dt (frame-invariant, no rotation needed).

    Two refinements over the naive version:
      - movement magnitude is scaled by the human's actual speed (clipped to [0,1] at
        speed_ref), instead of always being a unit vector — so the robot slows/stops
        with the human and stops over-driving the gait.
      - facing is low-pass filtered (EMA on the vector, wrap-safe) to remove the raw
        head-yaw jitter that made the base turn erratically.

    Returns: (movement, facing, speed, height)
    """
    delta_umi = curr_head_pos - prev_head_pos
    horiz_speed = float(np.linalg.norm(delta_umi[:2]) / dt) if dt > 0 else 0.0

    # Facing via delta-yaw relative to initial head orientation.
    # R_head.apply([0,0,1]) = camera-Z axis in UMI world = looking direction.
    # arctan2(y,x) gives the horizontal yaw of that direction in UMI world.
    head_rot = sRot.from_quat(curr_head_q_xyzw.astype(np.float64))
    looking = head_rot.apply([0.0, 0.0, 1.0])
    current_yaw_umi = float(np.arctan2(looking[1], looking[0]))

    delta_yaw = current_yaw_umi - calib["initial_head_yaw_umi"]
    delta_yaw = float(((delta_yaw + np.pi) % (2.0 * np.pi)) - np.pi)

    raw_facing = np.array([np.cos(delta_yaw), np.sin(delta_yaw), 0.0])

    # EMA low-pass on the facing VECTOR (wrap-safe), then renormalize. State lives in
    # calib, which is recreated per episode so smoothing resets cleanly each episode.
    prev_facing = calib.get("_smoothed_facing")
    if prev_facing is None or facing_alpha >= 1.0:
        sm = raw_facing
    else:
        sm = facing_alpha * raw_facing + (1.0 - facing_alpha) * prev_facing
        n = float(np.linalg.norm(sm[:2]))
        sm = np.array([sm[0] / n, sm[1] / n, 0.0]) if n > 1e-6 else raw_facing
    calib["_smoothed_facing"] = sm
    facing = sm.tolist()

    # Movement DIRECTION. By default the robot walks where it FACES (gaze). With
    # walk_direction, steer the legs along the human's ACTUAL displacement direction —
    # the human walks up to ~50° off their gaze (navigate-to-target), so gaze-only steering
    # makes the robot under-turn. facing (head/torso target) stays gaze-based either way.
    # robot_walk_yaw = walk_yaw_umi - initial_gaze_yaw (aligns frame-0 gaze to robot +X).
    move_dir = np.array(facing, dtype=np.float64)
    if walk_direction and horiz_speed > speed_deadzone:
        walk_yaw = float(np.arctan2(delta_umi[1], delta_umi[0]) - calib["initial_head_yaw_umi"])
        raw_dir = np.array([np.cos(walk_yaw), np.sin(walk_yaw), 0.0])
        prev_dir = calib.get("_smoothed_movedir")
        if prev_dir is None or facing_alpha >= 1.0:
            md = raw_dir
        else:
            md = facing_alpha * raw_dir + (1.0 - facing_alpha) * prev_dir
            n = float(np.linalg.norm(md[:2]))
            md = np.array([md[0] / n, md[1] / n, 0.0]) if n > 1e-6 else raw_dir
        calib["_smoothed_movedir"] = md
        move_dir = md

    # Normalized human-speed magnitude (0..1), the analog of pico's joystick `mag`.
    if horiz_speed <= speed_deadzone or speed_ref <= 0:
        mag = 0.0
    else:
        mag = float(np.clip(horiz_speed / speed_ref, 0.0, 1.0))

    if match_speed:
        # SLOW_WALK semantics (pico_manager_thread_server.py:1740-1741): send an EXPLICIT
        # speed so the robot's pace tracks the human's instead of the WBC default.
        # CRITICAL: gate idle/move on the smoothed HUMAN SPEED (m/s) vs speed_deadzone —
        # the human moves in bursts (~half the frames are near-stationary), and if we don't
        # idle through the pauses the robot strolls the whole episode and over-travels ~3x.
        v_sm = speed_alpha * horiz_speed + (1.0 - speed_alpha) * calib.get("_smoothed_speed", 0.0)
        calib["_smoothed_speed"] = v_sm
        if v_sm <= speed_deadzone:
            movement = [0.0, 0.0, 0.0]
            speed = -1.0
        else:
            m = float(np.clip(v_sm / speed_ref, 0.0, 1.0))
            speed = float(SLOW_WALK_FLOOR + (speed_max - SLOW_WALK_FLOOR) * m)  # floor .. speed_max
            movement = [move_dir[0] * m, move_dir[1] * m, 0.0]
    else:
        # WALK semantics: speed=-1 (WBC default pace); magnitude only scales the command.
        movement = [move_dir[0] * mag, move_dir[1] * mag, 0.0]
        speed = -1.0

    # height is ALWAYS the default sentinel -1.0 ("keep standing height"), matching
    # pico_manager VR_3PT (pico_manager_thread_server.py:1807). The UMI head Z is a
    # tiny SLAM-relative offset (~0), NOT an absolute height — sending it commands a crouch.
    height = -1.0

    return movement, facing, speed, height


# ---------------------------------------------------------------------------
# ZMQ publishers
# ---------------------------------------------------------------------------

def build_manager_state_msg(stream_mode: int, toggle_record: bool = False) -> bytes:
    """Pack a manager_state message to control the data exporter."""
    data = {
        "stream_mode":           np.array([stream_mode],       dtype=np.int32),
        "toggle_data_collection": np.array([toggle_record],    dtype=np.bool_),
        "toggle_data_abort":      np.array([False],            dtype=np.bool_),
    }
    return pack_pose_message(data, topic="manager_state")


def build_camera_msg(frame_rgb: np.ndarray, timestamp: float) -> bytes:
    """Pack an ego_view frame in the same msgpack format as the camera bridge."""
    _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    msg = {
        "timestamps": {"ego_view": timestamp},
        "images":     {"ego_view": jpeg.tobytes()},
    }
    return msgpack.packb(msg, use_bin_type=True)


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

class VideoReader:
    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        self._prev = -1

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx != self._prev + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        self._prev = frame_idx
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __del__(self):
        self._cap.release()


def find_video(icra_root: Path, camera: str, ep_meta: pd.Series) -> str | None:
    chunk_col = f"videos/observation.images.{camera}/chunk_index"
    file_col  = f"videos/observation.images.{camera}/file_index"
    if chunk_col not in ep_meta.index:
        return None
    chunk = int(ep_meta[chunk_col])
    file_ = int(ep_meta[file_col])
    p = icra_root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file_:03d}.mp4"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Episode replay
# ---------------------------------------------------------------------------

def replay_episode(
    ep_idx:     int,
    ep_df:      pd.DataFrame,
    ep_meta:    pd.Series,
    tasks_df:   pd.DataFrame,
    icra_root:  Path,
    sonic_pub:  zmq.Socket,   # port 5556 — planner + manager_state
    camera_pub: zmq.Socket,   # port 5555 — ego_view
    fps:        float,
    ego_camera: str,
    pause_between_episodes: float = 2.0,
    no_locomotion: bool = False,
    speed_ref: float = 0.3,
    facing_alpha: float = 0.2,
    match_speed: bool = False,
    speed_max: float = 0.5,
    speed_deadzone: float = 0.06,
    no_camera: bool = False,
    walk_direction: bool = False,
    head_stabilize: bool = True,
):
    """Replay one ICRA episode into the data collection pipeline.

    no_locomotion: force the robot to stand still (mode=IDLE, zero movement) so the
    arm/vr_3pt tracking can be verified in isolation without the base wandering.
    """
    task_idx  = int(ep_df["task_index"].iloc[0])
    task_desc = tasks_df.index[task_idx] if task_idx < len(tasks_df) else "manipulation task"
    n_frames  = len(ep_df)
    dt        = 1.0 / fps

    print(f"\n[Episode {ep_idx:4d}] {task_desc[:60]}  ({n_frames} frames @ {fps} Hz)")

    # Open video
    video_path = find_video(icra_root, ego_camera, ep_meta)
    video = VideoReader(video_path) if video_path else None
    if video is None:
        print(f"  Warning: {ego_camera} video not found — sending blank frames")

    # Calibration from first frame — neck orientation + wrist position offset
    first = ep_df.iloc[0]
    calib = calibrate_from_first_frame(
        lw_pos_cam = np.array(first["observation.pose.left_gripper"][:3],  dtype=np.float64),
        lw_q_xyzw  = np.array(first["observation.pose.left_gripper"][3:7], dtype=np.float64),
        rw_pos_cam = np.array(first["observation.pose.right_gripper"][:3], dtype=np.float64),
        rw_q_xyzw  = np.array(first["observation.pose.right_gripper"][3:7],dtype=np.float64),
        nk_q_xyzw  = np.array(first["observation.pose.head"][3:7],         dtype=np.float64),
        head_stabilize = head_stabilize,
    )

    # Set stream_mode=5 (PLANNER_VR_3PT)
    sonic_pub.send(build_manager_state_msg(stream_mode=5))
    time.sleep(0.1)

    # Signal start of recording
    sonic_pub.send(build_manager_state_msg(stream_mode=5, toggle_record=True))
    time.sleep(0.05)

    prev_head_pos = np.array(ep_df.iloc[0]["observation.pose.head"][:3], dtype=np.float64)
    frame_start   = time.monotonic()

    for fi, (_, row) in enumerate(ep_df.iterrows()):
        target_t = frame_start + fi * dt

        # --- Parse UMI poses ---
        lw_pos_cam = np.array(row["observation.pose.left_gripper"][:3], dtype=np.float64)
        lw_q_xyzw  = np.array(row["observation.pose.left_gripper"][3:7], dtype=np.float64)
        rw_pos_cam = np.array(row["observation.pose.right_gripper"][:3], dtype=np.float64)
        rw_q_xyzw  = np.array(row["observation.pose.right_gripper"][3:7], dtype=np.float64)
        nk_q_xyzw  = np.array(row["observation.pose.head"][3:7], dtype=np.float64)
        curr_head_pos = np.array(row["observation.pose.head"][:3], dtype=np.float64)

        # --- Convert to Sonic 3pt ---
        vr_3pt_pos, vr_3pt_orn = compute_3pt_from_umi(
            lw_pos_cam, lw_q_xyzw,
            rw_pos_cam, rw_q_xyzw,
            nk_q_xyzw,
            calib=calib,
            head_stabilize=head_stabilize,
        )

        # --- Locomotion from head world trajectory ---
        movement, facing, speed, height = head_delta_to_planner(
            prev_head_pos, curr_head_pos, nk_q_xyzw, dt, calib,
            speed_ref=speed_ref, facing_alpha=facing_alpha,
            match_speed=match_speed, speed_max=speed_max,
            speed_deadzone=speed_deadzone, walk_direction=walk_direction,
        )
        prev_head_pos = curr_head_pos

        # --- Gripper commands ---
        umi_state = np.array(row["observation.state"], dtype=np.float32)
        left_gripper_cmd  = float(umi_state[0])  # left_rad
        right_gripper_cmd = float(umi_state[2])  # right_rad

        # --- Send planner message (replaces pico_manager) ---
        # LocomotionMode: IDLE=0, SLOW_WALK=1, WALK=2
        if no_locomotion:
            # Stand still: isolate arm/vr_3pt tracking from base motion for debugging.
            movement = [0.0, 0.0, 0.0]
            facing = [1.0, 0.0, 0.0]
        horiz_mag = float(np.linalg.norm(movement[:2]))
        # match_speed sends an explicit speed → SLOW_WALK (1); else WALK (2) at WBC default pace.
        walk_mode = 1 if match_speed else 2
        loco_mode = walk_mode if horiz_mag > 0.05 else 0
        planner_msg = build_planner_message(
            mode       = loco_mode,
            movement   = movement,
            facing     = facing,
            speed      = speed,
            height     = height,
            vr_3pt_position    = vr_3pt_pos.tolist(),
            vr_3pt_orientation = vr_3pt_orn.tolist(),
            left_gripper_cmd   = left_gripper_cmd,
            right_gripper_cmd  = right_gripper_cmd,
        )
        sonic_pub.send(planner_msg)

        # --- Send camera frame (replaces camera bridge) ---
        # The per-frame video decode + JPEG encode is the loop's bottleneck. When it can't
        # keep up, the replay runs slower than realtime, which time-stretches the episode and
        # makes the robot over-travel (the deploy integrates each velocity command for longer
        # than the data's dt). Skip it for locomotion testing / when images aren't needed.
        if not no_camera:
            if video is not None:
                try:
                    frame = video.read(fi)
                    frame = cv2.resize(frame, (640, 480))
                except Exception:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            camera_pub.send(build_camera_msg(frame, time.time()))

        # --- Pace to target FPS ---
        now = time.monotonic()
        sleep = target_t - now
        if sleep > 0:
            time.sleep(sleep)

    # Signal end of recording
    sonic_pub.send(build_manager_state_msg(stream_mode=5, toggle_record=True))
    elapsed = time.monotonic() - frame_start
    eff_fps = n_frames / elapsed if elapsed > 0 else 0.0
    realtime = "REALTIME" if eff_fps >= fps * 0.9 else f"SLOW ({eff_fps/fps:.2f}x — robot will over-travel!)"
    print(f"  Episode {ep_idx} complete in {elapsed:.1f}s — effective {eff_fps:.1f} fps [{realtime}]")
    print(f"  waiting {pause_between_episodes}s before next")
    time.sleep(pause_between_episodes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_episodes(s: str, n_total: int) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), min(int(hi) + 1, n_total)))
    elif "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(description="Replay ICRA UMI dataset into GR00T data collection pipeline")
    parser.add_argument("--icra-root",  required=True,           help="Path to ~/icra-dataset")
    parser.add_argument("--episodes",   default=None,            help="Episode range e.g. '0-9' or '0,3,5'")
    parser.add_argument("--fps",        type=float, default=30,  help="Replay FPS (default 30 to match ICRA)")
    parser.add_argument("--sonic-host", default="localhost",      help="Host running data collection pipeline")
    parser.add_argument("--sonic-port", type=int, default=5556,  help="ZMQ port for planner messages")
    parser.add_argument("--camera-port",type=int, default=5555,  help="ZMQ port for camera frames")
    parser.add_argument("--ego-camera", default="camera_depth_head", help="ICRA camera to use as ego_view")
    parser.add_argument("--pause",      type=float, default=2.0, help="Pause between episodes (seconds)")
    parser.add_argument("--no-locomotion", action="store_true",
                        help="Force robot to stand still (IDLE, zero movement) — isolates arm/vr_3pt tracking for sim debugging")
    parser.add_argument("--speed-ref", type=float, default=0.3,
                        help="Human speed (m/s) that maps to full movement magnitude; lower = robot walks faster for same human motion")
    parser.add_argument("--facing-smooth", type=float, default=0.2,
                        help="EMA weight for facing low-pass (0..1; lower = smoother/laggier, 1.0 = off)")
    parser.add_argument("--match-speed", action="store_true",
                        help="SLOW_WALK mode: send explicit speed ~ human speed so walked distance matches the demo (else WALK at WBC default pace)")
    parser.add_argument("--speed-max", type=float, default=0.5,
                        help="Upper end of the SLOW_WALK speed ramp in m/s (pico uses 0.6)")
    parser.add_argument("--loco-deadzone", type=float, default=0.06,
                        help="Smoothed human speed (m/s) below which the robot idles; raise to walk less (shorter distance)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip per-frame video decode/encode/send. Removes the loop bottleneck so replay runs at realtime — required for correct locomotion timing when images aren't needed")
    parser.add_argument("--walk-direction", action="store_true",
                        help="Steer the base along the human's actual walk direction (head displacement), not gaze — the human walks up to ~50° off gaze, so gaze steering under-turns")
    parser.add_argument("--no-head-stabilize", action="store_true",
                        help="Revert to the legacy constant-Q hand mapping (frame-0 head tilt only). By default the live head quat removes per-frame head rotation so the hand reaches forward correctly when the operator looks down into a bin")
    args = parser.parse_args()

    icra_root = Path(args.icra_root)

    # --- Load metadata ---
    print("[Init] Loading ICRA dataset...")
    all_df   = pd.read_parquet(icra_root / "data" / "chunk-000" / "file-000.parquet")
    ep_meta  = pd.read_parquet(icra_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(icra_root / "meta" / "tasks.parquet")
    all_ep_ids = sorted(all_df["episode_index"].unique().tolist())
    print(f"  {len(all_df)} frames, {len(all_ep_ids)} episodes, {len(tasks_df)} tasks")

    episode_ids = parse_episodes(args.episodes, len(all_ep_ids)) if args.episodes else all_ep_ids
    print(f"  Replaying {len(episode_ids)} episodes")

    # --- ZMQ setup ---
    ctx = zmq.Context()

    sonic_pub = ctx.socket(zmq.PUB)
    sonic_pub.bind(f"tcp://*:{args.sonic_port}")
    print(f"[ZMQ] Bound planner PUB on port {args.sonic_port}")

    camera_pub = ctx.socket(zmq.PUB)
    camera_pub.bind(f"tcp://*:{args.camera_port}")
    print(f"[ZMQ] Bound camera PUB on port {args.camera_port}")

    # Give subscribers time to connect
    print("[Init] Waiting 1s for subscribers to connect...")
    time.sleep(1.0)

    # --- Startup: transition robot OFF → PLANNER (standing) ---
    # ZMQ PUB/SUB does not buffer — if the deploy SUB hasn't connected yet when we
    # send, the message is dropped. Keep resending every 200ms for 5s so the deploy
    # catches it whenever its initialization finishes.
    print("[Init] Sending start command (OFF → PLANNER) — retrying for 5s...")
    t_end = time.monotonic() + 5.0
    while time.monotonic() < t_end:
        sonic_pub.send(build_command_message(start=True, stop=False, planner=True))
        time.sleep(0.2)
    print("[Init] Waiting 3s for robot to stand up...")
    time.sleep(3.0)

    # --- Replay ---
    print(f"\n[Replay] Starting...\n")
    for ep_idx in episode_ids:
        ep_df = all_df[all_df["episode_index"] == ep_idx].sort_values("frame_index")
        if len(ep_df) == 0:
            print(f"  Episode {ep_idx}: no frames, skipping")
            continue
        ep_meta_row = ep_meta.iloc[ep_idx] if ep_idx < len(ep_meta) else pd.Series()
        replay_episode(
            ep_idx=ep_idx,
            ep_df=ep_df,
            ep_meta=ep_meta_row,
            tasks_df=tasks_df,
            icra_root=icra_root,
            sonic_pub=sonic_pub,
            camera_pub=camera_pub,
            fps=args.fps,
            ego_camera=args.ego_camera,
            pause_between_episodes=args.pause,
            no_locomotion=args.no_locomotion,
            speed_ref=args.speed_ref,
            facing_alpha=args.facing_smooth,
            match_speed=args.match_speed,
            speed_max=args.speed_max,
            speed_deadzone=args.loco_deadzone,
            no_camera=args.no_camera,
            walk_direction=args.walk_direction,
            head_stabilize=not args.no_head_stabilize,
        )

    print("\n[Done] All episodes replayed.")
    sonic_pub.close()
    camera_pub.close()
    ctx.term()


if __name__ == "__main__":
    main()
