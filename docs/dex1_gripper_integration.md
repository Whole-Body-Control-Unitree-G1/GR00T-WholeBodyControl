# Dex1-1 Parallel Gripper Integration

## Overview

This documents the Unitree Dex1-1 parallel gripper integration on branch `kjs-dex_1_1`.
The Dex1-1 replaces the 7-DOF Dex3 dexterous hand. It is a 1-DOF parallel gripper controlled
by a single motor, mapped from the PICO VR controller trigger buttons.

## Hardware

- **Gripper:** Unitree Dex1-1 (1-DOF parallel gripper, left + right)
- **Max angle:** 5.5 rad (fully open → fully closed)
- **Controller:** kp=5.0, kd=0.05

## Architecture

```
PICO VR Headset (trigger buttons)
        │
        ▼
pico_manager_thread_server.py  (Laptop)
  - reads left_trigger / right_trigger [0, 1] via XRoboToolkit
  - publishes rt/dex1/left/cmd + rt/dex1/right/cmd  (DDS)
  - streams left_gripper_cmd / right_gripper_cmd in ZMQ data
        │
        │  DDS (unitree_sdk2py over network interface)
        ▼
dex1_1_service  (Robot / PC2)
  - low-level motor driver bridge
  - publishes rt/dex1/left/state + rt/dex1/right/state (DDS)
        │
        ▼
run_data_exporter.py  (Laptop)
  - polls Dex1GripperReader for state
  - records observation + action into GR00T dataset
```

## DDS Topics

| Topic                  | Type         | Direction          | Description              |
|------------------------|--------------|--------------------|--------------------------|
| `rt/dex1/left/cmd`     | `MotorCmds_` | Laptop → Robot     | Left gripper position cmd |
| `rt/dex1/right/cmd`    | `MotorCmds_` | Laptop → Robot     | Right gripper position cmd|
| `rt/dex1/left/state`   | `MotorStates_` | Robot → Laptop   | Left gripper state        |
| `rt/dex1/right/state`  | `MotorStates_` | Robot → Laptop   | Right gripper state       |

## Run Order

### 1. Robot — start dex1_1_service

On the robot (G1 Orin / PC2), run the Unitree dex1_1_service binary. This bridges
the physical gripper hardware to DDS.

```bash
# On the robot
./dex1_1_service
```

Source: https://github.com/unitreerobotics/dex1_1_service

### 2. Laptop — find the network interface

Find the interface connected to the robot's network:

```bash
ip link show
# or
ifconfig
```

Use the interface name that is on the same subnet as the robot (e.g. `eth0`, `wlan0`, `enp3s0`).

### 3. Laptop — start pico_manager (teleop + gripper control)

```bash
cd ~/wbcG1/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --vis_vr3pt \
    --dex1_interface <your_network_interface>
```

- Trigger buttons on PICO → gripper position commands sent in real time
- While holding the **left menu button**, gripper commands are suppressed (for repositioning without closing)
- Gripper commands are also forwarded into the ZMQ data stream for recording

### 4. Laptop — start data exporter (recording)

```bash
cd ~/wbcG1/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --with_dex1_grippers \
    --dex1_network_interface <your_network_interface>
```

The `--with_dex1_grippers` flag enables `Dex1GripperReader` for both grippers.
If `unitree_sdk2py` is not available it degrades gracefully (records zeros with a warning).

## Dataset Schema

### Why 1D instead of 7D

The original GR00T VLA action space assumed a 7-DOF Dex3 dexterous hand (`rt/dex3/*/cmd`).
The Dex1-1 has only 1 motor per hand, so `features_sonic_vla.py` was updated to match:

| Key                              | Shape | Description                         |
|----------------------------------|-------|-------------------------------------|
| `observation.left_gripper_state` | (3,)  | [q, dq, tau_est] of left gripper    |
| `observation.right_gripper_state`| (3,)  | [q, dq, tau_est] of right gripper   |
| `action.left_gripper_cmd`        | (1,)  | Trigger value [0, 1] → position cmd |
| `action.right_gripper_cmd`       | (1,)  | Trigger value [0, 1] → position cmd |

**Note:** Training datasets from Dex3 (7D) hands are not action-space compatible.

### Modality config keys

In `features_sonic_vla.py`, the gripper entries use:
- `observation` modality: keys `left_gripper_state`, `right_gripper_state` (indices 0–3)
- `action` modality: keys `left_gripper_cmd`, `right_gripper_cmd` (indices 0–1)

## Key Files

| File | Role |
|------|------|
| `gear_sonic/utils/teleop/dex1_gripper_reader.py` | DDS subscriber for gripper state |
| `gear_sonic/utils/teleop/dex1_gripper_sender.py` | DDS publisher for gripper commands |
| `decoupled_wbc/control/envs/g1/g1_dex1_gripper.py` | Gym Env wrapper (observe + queue_action) |
| `decoupled_wbc/control/envs/g1/utils/command_sender.py` | `Dex1CommandSender` class |
| `decoupled_wbc/control/envs/g1/utils/state_processor.py` | `Dex1StateProcessor` class |
| `gear_sonic/data/features_sonic_vla.py` | Dataset feature schema (updated for 1D) |
| `gear_sonic/scripts/run_data_exporter.py` | Data collection (add `--with_dex1_grippers`) |
| `gear_sonic/scripts/pico_manager_thread_server.py` | Teleop server (add `--dex1_interface`) |
