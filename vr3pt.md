# VR 3-Point Pose Pipeline

How PICO VR tracking data flows from headset → coordinate transforms → Sonic encoder → G1 joint commands during PLANNER_VR_3PT (stream_mode=5) data collection.

---

## 1. What the PICO Captures

The PICO headset runs XRoboToolkit's body-tracking app, which runs an on-device SMPL body estimation model. Every frame it provides:

```
xrt.get_body_joints_pose()  →  shape (24, 7)
```

Each of the 24 SMPL joints is: `[x, y, z, qx, qy, qz, qw]`  
Quaternion convention: **scalar-last** `[x, y, z, w]`  
Coordinate frame: **Unity** — Y-up, left-handed, Z-forward

```
Unity frame:
  X  →  right
  Y  ↑  up
  Z  ·  forward (into screen)
```

The 4 joints we care about:

| SMPL index | Body part | Why |
|---|---|---|
| 0 | Root / Pelvis | Reference origin |
| 12 | Neck | More stable than Head (joint 15) for body orientation — less "looking around" noise |
| 22 | Left Wrist | Left end-effector |
| 23 | Right Wrist | Right end-effector |

---

## 2. Unity → Robot Frame Transform (`_compute_rel_transform`)

The robot uses a **Z-up, right-handed, X-forward** frame:

```
Robot frame:
  X  ·  forward
  Y  ←  left
  Z  ↑  up
```

The mapping from Unity to Robot is:

```
Robot_x = -Unity_x
Robot_y =  Unity_z
Robot_z =  Unity_y
```

As a matrix `Q`:
```
Q = [[-1, 0, 0],
     [ 0, 0, 1],
     [ 0, 1, 0]]
```

This is applied to every joint's position AND orientation (rotation matrices transform as `Q @ R @ Q.T`).

After this step, all 24 joints are in robot frame with **scalar-first** quaternions `[w, x, y, z]`.

---

## 3. Rotation Offsets per Keypoint

SMPL joint frames don't naturally align with the G1's link frames. Fixed offsets are applied per keypoint (extrinsic rotations about the original frame's axes):

| Keypoint | Offset | Purpose |
|---|---|---|
| Root (0) | yaw −90° | Align SMPL pelvis X-forward to robot convention |
| Left Wrist (22) | roll +90° | Align SMPL wrist to G1 left wrist frame |
| Right Wrist (23) | roll −90°, yaw +180° | Align SMPL wrist to G1 right wrist frame (mirrored) |
| Neck (12) | yaw −90° | Align SMPL neck to robot convention |

These are **post-multiplied**: `new_rot = original_rot * OFFSET` (intrinsic).

---

## 4. Make Everything Root-Relative

All three non-root keypoints (L-Wrist, R-Wrist, Neck) are expressed **relative to the Root (pelvis)**:

```python
# Position: subtract root, rotate by inverse of root orientation
keypoint_pos_rel = inv(root_rot).apply(keypoint_pos - root_pos)

# Orientation: relative rotation
keypoint_rot_rel = inv(root_rot) * keypoint_rot
```

**After this step**, the root is implicitly at `(0, 0, 0)` with identity orientation. Everything is in the root's local coordinate frame — i.e., if the operator's pelvis moves or rotates in the real world, the keypoint positions don't change (they're body-relative, not world-relative).

Output of `_process_3pt_pose()`:

```
shape (3, 7) — [L-Wrist, R-Wrist, Neck]
Each row: [x, y, z, qw, qx, qy, qz]  ← scalar-first quaternion
Frame: root-relative, robot coordinate convention
```

---

## 5. Calibration (`ThreePointPose`)

Raw VR poses have two problems:
1. The neck may not be upright at the start — any initial tilt bleeds into wrist positions
2. The VR wrist frames may not match the G1's actual wrist link frames at rest pose

Calibration is captured once at session start (operator in neutral standing pose):

### Step 1 — Neck orientation zeroing

The initial neck quaternion is captured and its inverse is stored:
```
calib_neck_inv = inv(initial_neck_rot)
```

Every subsequent frame applies this to the neck orientation:
```
calibrated_neck_rot = calib_neck_inv * current_neck_rot
```
Result: when the operator stands upright, neck is identity (pointing straight up). Head tilts/turns are now relative to this neutral.

### Step 2 — Wrist position alignment to G1 FK

1. Apply `calib_neck_inv` to both wrist positions/orientations (remove the initial neck tilt)
2. Compute the G1's actual left/right wrist positions via **Forward Kinematics** at the zero-pose (all joint angles = 0)
3. Compute offset = `VR_wrist_corrected − FK_wrist_position`

Every frame:
```
calibrated_lwrist_pos = calib_neck_inv.apply(raw_lwrist_pos) - lwrist_offset
calibrated_rwrist_pos = calib_neck_inv.apply(raw_rwrist_pos) - rwrist_offset
calibrated_lwrist_rot = lwrist_rot_offset * (calib_neck_inv * raw_lwrist_rot)
calibrated_rwrist_rot = rwrist_rot_offset * (calib_neck_inv * raw_rwrist_rot)
```

### Step 3 — Neck position via kinematic chain

The neck's position is NOT taken from SMPL directly (too noisy). Instead it's reconstructed from the calibrated neck orientation:

```python
TORSO_LINK_OFFSET_Z = 0.05   # meters: root → torso_link (along Z)
NECK_LINK_LENGTH    = 0.35   # meters: torso_link → neck (along neck's local Z)

neck_z_axis = calibrated_neck_rot.apply([0, 0, 1])
neck_pos = [0, 0, TORSO_LINK_OFFSET_Z] + NECK_LINK_LENGTH * neck_z_axis
```

So the neck's XY position is entirely determined by where the neck is pointing — if you tilt forward, the neck position moves forward. This encodes **waist height** implicitly: tilting the body down lowers the neck Z.

---

## 6. Final Calibrated 3-Point Pose

After calibration, the output is shape `(3, 7)`:

```
Row 0: L-Wrist  [x, y, z, qw, qx, qy, qz]
Row 1: R-Wrist  [x, y, z, qw, qx, qy, qz]
Row 2: Neck     [x, y, z, qw, qx, qy, qz]

Frame: root-relative, robot Z-up convention
Units: meters
```

Typical values for a standing operator with arms at sides:
- Neck: `[0, 0, ~0.38]` (38 cm above pelvis, pointing up)
- L-Wrist: `[-0.1, +0.2, 0.0]` (slightly left, slightly out, at hip height)
- R-Wrist: `[-0.1, -0.2, 0.0]` (mirrored)

---

## 7. Packing and Sending over ZMQ

The 3-point pose is packed into a `planner` ZMQ message on port 5556.

Positions are sent as-is (9 floats total):
```
vr_position (9,) float32 = [lwrist_xyz, rwrist_xyz, neck_xyz]
```

Orientations: quaternions are converted to **6D rotation** (first two columns of the 3×3 rotation matrix, flattened):
```
vr_orientation (12,) float32 = [lwrist_qwxyz, rwrist_qwxyz, neck_qwxyz]
```
*(Note: stored as quaternions in the wire format; the data exporter converts to 6D on write)*

The data exporter (`run_data_exporter.py`) receives this and stores:
```
teleop.vr_3pt_position   shape (9,)  float32  — raw root-relative positions
teleop.vr_3pt_orientation shape (18,) float32  — 6D rotation (2 cols of 3×3 matrix per keypoint)
```

6D rotation layout per keypoint:
```
[r00, r10, r01, r11, r02, r12]
where rij = element [i,j] of the 3×3 rotation matrix
= first two COLUMNS of R, each written column-major
```

---

## 8. Sonic Encoder — 3PT → Motion Token

The Sonic ONNX encoder takes a concatenated observation vector and outputs a 64-dim motion token:

```
Input (concatenated float32 vector):
  encoder_mode                          (1,)   — mode ID, e.g. 5 for VR3PT
  motion_joint_positions_lowerbody      (N,)   — lower body joint angles, 10 frames × stride 5
  motion_joint_velocities_lowerbody     (N,)   — lower body joint velocities, same windowing
  vr_3pt_position                       (9,)   — calibrated [lwrist, rwrist, neck] positions
  vr_3pt_orientation                    (18,)  — 6D rotations for each keypoint
  motion_anchor_orientation             (6,)   — base body orientation as 6D rotation

Output:
  motion_token  (64,) float32  — latent whole-body motion representation
```

The **lower body joints** matter because Sonic's motion policy jointly optimises upper and lower body. In VR3PT mode, the lower body naturally follows from the robot's current state (it's standing still or walking), so the encoder uses the last 10 measured lower body joint positions to "condition" the upper body prediction — it knows where the legs are before deciding where to put the arms.

The **anchor orientation** is the 6D representation of the base/pelvis orientation, which tells the encoder how the robot body is tilted relative to gravity.

---

## 9. Sonic Decoder — Motion Token → Joint Targets

The motion token feeds into the Sonic decoder (ONNX) which outputs full-body joint targets for the G1:

```
motion_token (64,) + current_state → joint_targets (43,)
```

The decoder runs at 50 Hz via the C++ deploy loop (`gear_sonic_deploy`). GR00T predicts motion tokens at 10 Hz (horizon=40, 40 tokens × 0.02s each), and the decoder consumes them sequentially at 50 Hz.

**Key insight**: GR00T never predicts joint angles directly. It predicts *what Sonic's encoder would have produced for this motion* — i.e., it predicts in the latent space of the motion policy. Sonic then handles the actual whole-body control.

---

## 10. Full Data Flow Summary

```
PICO headset
│  xrt.get_body_joints_pose()
│  24 SMPL joints, Unity frame [x,y,z,qx,qy,qz,qw], scalar-last
│
▼
_process_3pt_pose()
│  1. All 24 joints: Unity → Robot frame  (Q matrix)
│  2. Extract joints 0, 12, 22, 23
│  3. Apply per-joint rotation OFFSETS
│  4. Make all 3 keypoints root-relative
│  Output: (3,7) root-relative, robot frame, scalar-first quat
│
▼
ThreePointPose._apply_calibration()
│  1. Neck orientation: apply inv(initial_neck)
│  2. Wrist positions: apply neck_inv + subtract FK offset
│  3. Wrist orientations: apply wrist_rot_offset
│  4. Neck position: recompute from kinematic chain (0.05 + 0.35 × neck_Z)
│  Output: (3,7) calibrated, pelvis-relative, robot frame
│
▼
ZMQ "planner" message (port 5556)
│  vr_position (9,) + vr_orientation (12,) — quaternion wire format
│
▼
run_data_exporter.py                         run_vla_inference.py (at inference)
│  Converts quat → 6D rotation               │  Same conversion
│  Stores to parquet:                         │
│    teleop.vr_3pt_position  (9,)             │
│    teleop.vr_3pt_orientation (18,)          │
│    action.motion_token (64,) ← from C++    │
│                                             │
▼                                             ▼
GR00T training                          Sonic ONNX Encoder
(learns: observation → motion_token)    vr_3pt + lower_body + anchor → token (64,)
                                              │
                                              ▼
                                        Sonic ONNX Decoder
                                        token → joint_targets (43,)
                                              │
                                              ▼
                                        G1 robot joints
```

---

## 11. Mapping UMI Dataset → This Pipeline

The UMI dataset captures the exact same 3-point information, in a different form:

| UMI field | Sonic equivalent | Notes |
|---|---|---|
| `observation.pose.left_gripper` xyz | `vr_3pt_position[0:3]` (lwrist) | Head-relative; add neck offset to get pelvis-relative |
| `observation.pose.right_gripper` xyz | `vr_3pt_position[3:6]` (rwrist) | Same |
| `observation.pose.head` xyz | `vr_3pt_position[6:9]` (neck) | Head is origin → `(0,0,0)` in head frame → add neck offset |
| `observation.pose.left_gripper` quat | `vr_3pt_orientation[0:6]` | Convert quat → 6D |
| `observation.pose.right_gripper` quat | `vr_3pt_orientation[6:12]` | Convert quat → 6D |
| `observation.pose.head` quat | `vr_3pt_orientation[12:18]` | Convert quat → 6D |
| Implicit (head Z below pelvis) | `planner_height` / neck Z | UMI head-Z encodes waist height naturally |

**The key coordinate transform**: UMI poses are in the **head/camera frame** (head = origin). Sonic needs **pelvis-relative** frame. Since the head is roughly 0.4 m above the pelvis in a standing posture:

```python
NECK_TO_PELVIS_OFFSET = np.array([0.0, 0.0, 0.40])  # meters, robot Z-up

# Convert UMI head-frame → pelvis-frame
neck_pos_pelvis   = NECK_TO_PELVIS_OFFSET                     # head is origin → (0,0,0) + offset
lwrist_pos_pelvis = umi_left_gripper_xyz  + NECK_TO_PELVIS_OFFSET
rwrist_pos_pelvis = umi_right_gripper_xyz + NECK_TO_PELVIS_OFFSET
```

**Coordinate convention**: UMI uses Z-up (same as robot frame). The gripper axes visible in the head-camera image confirm this — grippers are below the head (negative Z in head frame), which after adding the offset lands at the correct pelvis-relative height.

**Lower body**: UMI has no lower body joint data. For standing manipulation tasks, use a **canonical standing pose** (all lower body joints at zero) as the lower body history input to the Sonic encoder. This is valid because UMI tasks are tabletop manipulation — the operator is standing still.

**What you still need to do**:
1. Determine exact UMI quaternion convention (scalar-first `[w,x,y,z]` or scalar-last `[x,y,z,w]`)
2. Verify the neck Z offset (0.40 m is approximate — actual G1 pelvis-to-neck is closer to 0.38–0.42 m depending on posture)
3. Apply the same rotation OFFSETS that `_process_3pt_pose` applies to align wrist frames with G1 link conventions
4. Feed through the Sonic encoder with zero lower body history → get motion tokens per frame
5. Pair with UMI camera frames → GR00T training dataset
