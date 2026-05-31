# icra_to_groot.py — Implementation Notes

Converts the ICRA UMI dataset (LeRobot v3.0) into live ZMQ messages that the
GR00T/Sonic data collection pipeline accepts, as if a PICO VR operator were
teleoperating the robot in real time.

---

## Coordinate Frames

Three frames appear throughout this code. It's critical not to mix them up.

### UMI Camera Frame (gripper poses)
Gripper positions and orientations in the dataset are expressed **relative to
the head camera**, in camera convention:

```
X → right
Y ↓ down
Z · forward (into the scene)
```

Quaternions are **scalar-last** `[qx, qy, qz, qw]`.

### UMI World Frame (head pose)
The head position/orientation is in a **gravity-aligned world frame**:

```
X, Y → horizontal (arbitrary orientation relative to robot)
Z    ↑ up
```

This frame shares the Z-up convention with the robot, but is **rotated by an
unknown angle around Z** (roughly −100°) relative to the robot world frame.
You cannot use `Q_CAM_TO_ROBOT` to convert positions or orientations from this
frame into robot frame.

### Robot Frame (Sonic/G1)
```
X · forward
Y ← left
Z ↑ up
```

Quaternions are **scalar-first** `[qw, qx, qy, qz]`.

---

## The Q Matrix — Camera → Robot

```python
Q_CAM_TO_ROBOT = [[0,  0,  1],
                  [-1, 0,  0],
                  [0, -1,  0]]
```

Mapping:
```
Robot X =  Camera Z   (camera forward  → robot forward)
Robot Y = -Camera X   (camera right    → robot left)
Robot Z = -Camera Y   (camera down     → robot up)
```

**Where Q is valid:** gripper positions and orientations from the UMI dataset,
because these are expressed in the head camera frame.

**Where Q is NOT valid:** the head world-frame position or the head quaternion
when used to derive an absolute direction in the robot world. The head pose is
in a Z-up world frame (not camera convention), so applying Q to it produces
garbage XY components. Only the rotation-change (delta-yaw) from the head
quaternion can be used without knowing the world→robot alignment angle.

### Applying Q to positions
```python
robot_pos = Q_CAM_TO_ROBOT @ camera_pos
```

### Applying Q to rotations
For a rotation matrix R expressed in camera frame, the equivalent in robot
frame is:
```python
R_robot = Q @ R_cam  # NOT Q @ R @ Q.T — the UMI body frame is already
                     # aligned, only the reference frame changes
```
In code: `sRot.from_matrix(Q_CAM_TO_ROBOT @ R_cam.as_matrix())`

---

## Quaternion Convention Bookkeeping

| Source | Convention | scipy input |
|---|---|---|
| UMI dataset `[qx,qy,qz,qw]` | scalar-last | `sRot.from_quat([x,y,z,w])` directly |
| Sonic wire format `[qw,qx,qy,qz]` | scalar-first | `sRot.as_quat(scalar_first=True)` |
| scipy internal | scalar-last | always |

After `cam_quat_xyzw_to_robot_wxyz()` returns a scalar-first array `[w,x,y,z]`,
to feed it back into scipy you must reorder: `arr[[1,2,3,0]]` to get `[x,y,z,w]`.

---

## Arm Pose Pipeline (compute_3pt_from_umi)

The goal is to produce `vr_3pt_position` (9 floats) and `vr_3pt_orientation`
(12 floats, quaternions) in the Sonic planner format. Sonic expects positions
**relative to the pelvis**, in robot frame.

### Step 1 — Camera frame → Robot frame
Apply Q to each gripper position and rotation matrix.

```python
lw_pos_robot = Q @ lw_pos_cam
lw_rot_robot = sRot.from_matrix(Q @ lw_rot_cam.as_matrix())
```

### Step 2 — Rotation offsets (post-multiply)
These align the UMI wrist sensor axes with the G1 link conventions. For UMI
grippers (which are cameras, not SMPL joints), the offsets are identity — the
camera frame already aligns with what the G1 wrist expects after Q.

```python
OFFSETS = [identity, identity, yaw_neg90_deg]
new_rot = original_rot * OFFSET   # intrinsic / body-frame rotation
```

The neck offset (yaw −90°) converts the camera convention "Z-forward" into
the SMPL neck convention "X-forward" that the G1 kinematic chain expects.

### Step 3 — Neck calibration (pre-multiply)
At episode frame 0, the neck rotation is captured and inverted:
```python
calib_neck_inv = initial_neck_rot.inv()
```

Every subsequent frame:
```python
pos_corrected  = calib_neck_inv.apply(pos_in_cam_frame)
rot_corrected  = calib_neck_inv * rot_with_offset
```

This removes whatever tilt the person's head had at calibration time, so that
"neutral head = identity rotation". Positions are also de-rotated into the
neck's calibrated frame, keeping them head-relative but tilt-free.

### Step 4 — Neck position from kinematic chain
Rather than using the noisy SLAM position of the head, the neck's pelvis-relative
position is reconstructed from its orientation:

```python
neck_z_axis = calibrated_neck_rot.apply([0, 0, 1])
neck_pos = [0, 0, TORSO_LINK_OFFSET_Z] + NECK_LINK_LENGTH * neck_z_axis
#         = [0, 0, 0.05]              + 0.35 * neck_z_axis
```

At calibration frame, `calibrated_neck_rot = identity`, so `neck_z = [0,0,1]`
and `neck_pos = [0, 0, 0.40]` always — deterministic, no noise.

As the person tilts forward, `neck_z` tilts, and `neck_pos` shifts forward and
downward, encoding both torso tilt and height correctly.

### Step 5 — Wrist positions: head-relative → pelvis-relative
Gripper positions from the dataset are offsets from the head origin. Adding
`neck_pos` converts them to pelvis-relative:

```python
lw_pos_pelvis = neck_pos + lw_pos_neck_corrected - lw_pos_offset
```

`lw_pos_offset` is computed at calibration time so that frame 0 lands exactly
at the G1's FK rest position:

```python
lw_pos_offset = measured_pelvis_pos_at_frame0 - FK_LWRIST_POS
# FK_LWRIST_POS = [0.3798, +0.1237, 0.0952]  (from MuJoCo FK at zero pose)
# FK_RWRIST_POS = [0.3798, -0.1237, 0.0952]
```

### Step 6 — Wrist rotation calibration (pre-multiply)
Computed once at frame 0:
```python
lw_rot_offset = FK_WRIST_ROT * measured_rot_at_frame0.inv()
#             = identity * inv(measured)
```

Applied each frame (pre-multiply = world-frame correction):
```python
lw_rot_calibrated = lw_rot_offset * lw_rot_corrected
```

At frame 0 this gives identity (G1 rest pose). Subsequent frames give the
rotation delta from that rest pose.

---

## Locomotion Pipeline (head_delta_to_planner)

### The core problem
The UMI head pose is in a Z-up world frame that is **rotated by ~−100° around
Z** relative to the robot world frame. We don't know this angle exactly.
Therefore we cannot apply Q (designed for camera→robot) to the head orientation
and get a correct robot-frame facing direction.

### Solution: delta-yaw
At calibration (frame 0), record the head's looking direction in UMI world:

```python
head_rot_0 = sRot.from_quat(head_q_xyzw)          # scalar-last
looking_0  = head_rot_0.apply([0.0, 0.0, 1.0])    # camera-Z = looking dir in UMI world
initial_yaw_umi = arctan2(looking_0[1], looking_0[0])
```

Each subsequent frame, compute the **change** in yaw:

```python
looking_t    = head_rot_t.apply([0.0, 0.0, 1.0])
current_yaw  = arctan2(looking_t[1], looking_t[0])
delta_yaw    = wrap_to_pi(current_yaw - initial_yaw_umi)
```

Map this delta to robot frame, assuming the robot starts facing +X (forward):

```python
facing = [cos(delta_yaw), sin(delta_yaw), 0.0]
```

This correctly encodes left/right turns without needing to know the absolute
UMI-world → robot-world alignment.

### Speed
Horizontal speed is derived from the UMI world position delta magnitude —
this is frame-invariant (the norm doesn't depend on the unknown Z-rotation):

```python
horiz_speed = norm(delta_pos[:2]) / dt
```

### Locomotion mode
```python
loco_mode = WALK (2)  if horiz_speed > 0.02 m/s  else IDLE (0)
movement  = facing    if moving                   else [0, 0, 0]
speed     = -1.0      # let planner choose natural speed
```

---

## Calibration Summary

All of the following are computed from episode frame 0 and stored in `calib`:

| Key | What it stores | How used |
|---|---|---|
| `calib_neck_inv` | Inverse of frame-0 neck rotation (robot frame, post-offset) | Pre-multiplied every frame to zero initial neck tilt |
| `lw_pos_offset` | `measured_pelvis_pos_frame0 - FK_LWRIST_POS` | Subtracted from pelvis-relative wrist pos each frame |
| `rw_pos_offset` | Same for right wrist | Same |
| `lw_rot_offset` | `FK_WRIST_ROT * measured_rot_frame0.inv()` | Pre-multiplied every frame to align frame-0 → identity |
| `rw_rot_offset` | Same for right wrist | Same |
| `initial_head_yaw_umi` | `arctan2` of head looking direction in UMI world at frame 0 | Subtracted from current yaw each frame for delta-yaw locomotion |

---

## G1 FK Reference Values (zero pose, from MuJoCo)

These are the pelvis-relative wrist positions when all G1 joints are at zero,
computed by running MuJoCo FK on `g1_29dof_with_hand.xml`:

```
left_wrist_yaw_link position:  [0.1998, +0.1487, 0.0952]
right_wrist_yaw_link position: [0.1998, -0.1487, 0.0952]

Key frame offsets (from vr3pt_pose_visualizer.py):
  left_wrist:  [0.18, -0.025, 0.0]
  right_wrist: [0.18, +0.025, 0.0]

Final FK targets:
  FK_LWRIST_POS = [0.3798, +0.1237, 0.0952]
  FK_RWRIST_POS = [0.3798, -0.1237, 0.0952]
  FK_WRIST_ROT  = identity
```

---

## What Does NOT Work (Lessons Learned)

**Q applied to head world-frame position deltas** — the head position is in
UMI world (Z-up, unknown Z-rotation). Applying Q to `delta_pos` gives mostly
robot +Y (sideways) motion even when the person is walking forward. Use
delta-yaw for direction; use `norm(delta_pos[:2])` for speed.

**SMPL wrist rotation offsets on UMI grippers** — the original pico_manager
offsets (roll ±90°, yaw 180°) are designed for SMPL joint frames from the
PICO body tracker. UMI grippers are cameras mounted on robot arms, not SMPL
joints. Applying SMPL offsets causes the arms to point in completely wrong
directions. Use identity offsets for both wrists.

**Hardcoded FK positions** — the initial guess of `[-0.1, 0.2, 0]` for FK
wrist positions was wrong. Always run MuJoCo FK from the actual URDF to get
true values.

**`nk_pos_calib` computed from `calib_neck_inv.apply([0,0,1])`** — at the
calibration frame, the calibrated neck IS identity, so `apply([0,0,1]) = [0,0,1]`
always. `nk_pos_calib` is a constant `[0, 0, 0.40]`. Computing it dynamically
introduced a subtle bug where the wrong vector was used.

**`speed = horiz_speed` passed to planner** — the planner's speed parameter
is poorly documented and sending the measured human speed caused the robot to
rock/fall. Always use `speed = -1.0` (planner default).

**`height = float(curr_head_pos[2])`** — the UMI head Z is a SLAM-relative
offset (~0.24 m), not an absolute head height. Sending it caused the planner
to command a deep crouch. Should be investigated further; for now `height`
is passed as the raw value and does not appear to destabilise the robot.
