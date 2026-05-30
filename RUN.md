# Run Guide

Full pipelines for teleoperated data collection and VLA inference with ZED stereo camera and Dex1-1 grippers.

## Prerequisites

- PICO VR headset paired in XRoboToolkit app
- `video_source.yml` pushed to PICO:
  ```bash
  adb push ~/video_source.yml /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
  ```
- Robot network interface name known (find with `ip link show`, e.g. `wlp0s20f3`)
- Robot Jetson NX time sync: `systemd-timesyncd` is enabled and runs automatically on boot — no manual action needed.
- Robot Jetson clocks set to max performance (one-time setup — persists across reboots):
  ```bash
  # Run once on the Jetson
  sudo tee /usr/local/bin/robot_startup.sh << 'EOF'
  #!/bin/bash
  /usr/bin/jetson_clocks
  EOF
  sudo chmod +x /usr/local/bin/robot_startup.sh

  sudo tee /etc/systemd/system/robot-startup.service << 'EOF'
  [Unit]
  Description=Robot startup — max clocks
  After=multi-user.target

  [Service]
  Type=oneshot
  ExecStart=/usr/local/bin/robot_startup.sh
  RemainAfterExit=yes

  [Install]
  WantedBy=multi-user.target
  EOF

  sudo systemctl enable robot-startup.service
  sudo systemctl start robot-startup.service
  ```
- `/dev/ttyUSB4` permissions set automatically on plug-in (one-time setup):
  ```bash
  sudo tee /etc/udev/rules.d/99-ttyusb.rules << 'EOF'
  KERNEL=="ttyUSB4", MODE="0666"
  EOF
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  ```
- Robot ROS2 middleware set to CycloneDDS for stable ZED camera publish rate (one-time setup):
  CycloneDDS delivers a stable **30 Hz** on the ZED HD720 topic over WiFi; FastDDS degrades to ~6 Hz
  after many episodes due to DDS buffer exhaustion.
  ```bash
  # Install CycloneDDS RMW
  sudo apt install ros-humble-rmw-cyclonedds-cpp

  # Merge tuning params into existing CycloneDDS config on the robot
  # (the file already pins the WiFi interface wlxfc23cd929b72)
  # Copy the pre-configured file from the laptop:
  scp ~/wbcG1/cyclonedds.xml unitree@192.168.123.164:~/.local/cyclone_config.xml

  # Persist network receive buffers (needed for large image messages)
  sudo tee /etc/sysctl.d/60-zed-buffers.conf << 'EOF'
  net.ipv4.ipfrag_time=3
  net.ipv4.ipfrag_high_thresh=134217728
  net.core.rmem_max=2147483647
  EOF
  sudo sysctl --system
  ```
  Then add to `~/.bashrc` on the robot (if not already present):
  ```bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///home/unitree/.local/cyclone_config.xml
  ```
  The source file `~/wbcG1/cyclonedds.xml` (on the laptop) is the canonical copy — edit it
  there and scp again if changes are needed.

---

## Robot (G1 Orin — 192.168.123.164)

### 1. ZED camera
```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm
```
Ensure `common_stereo.yaml` has `publish_stereo: true` and `zedm.yaml` has `grab_frame_rate: 60`.

### 2. ZMQ bridge
Publishes `ego_view` (640×480 RGB, for GR00T) and `stereo_view` (stereo, for PICO) on port 5555.
```bash
ros2 launch head_zmq_bridge zmq_bridge.launch.xml
```

### 3. Dex1-1 gripper service
```bash
./dex1_1_service
```

### 4. Head motor control + state bridge (only when using `--with-head`)
Starts the head motor controller and republishes the actual head joint state on ZMQ port 5558
so the laptop data exporter can record `observation.head_state` / `action.head_cmd`.
```bash
# Source the headControl workspace (built once with colcon in ~/headControl)
source ~/headControl/install/setup.bash

# Head motor controller
ros2 launch head_control head_control.launch.xml &

# Bridge node: reads VR head pose from laptop ZMQ (5556), commands head motors,
# re-publishes actual head/state on ZMQ port 5558.
# Replace <laptop_ip> with the laptop's LAN IP (e.g. 192.168.123.222).
ros2 run head_pico_bridge head_pico_bridge --ros-args -p zmq_host:=<laptop_ip>
```

---

## Laptop

### 4. XRoboToolkit PC service
```bash
bash /opt/apps/roboticsservice/runService.sh
```
Connect the PICO to the XRoboToolkit app and confirm it is paired.

### 5. PICO video stream
Streams the stereo ZED view to the PICO headset at 60fps.
```bash
source ~/wbcG1/GR00T-WholeBodyControl/.venv_teleop/bin/activate
python3 ~/wbcG1/headControl/src/headctrl/head_pico_bridge/head_pico_bridge/zed_pico_zmq.py
```
Then trigger the camera stream from the PICO XRoboToolkit app.

---

## Data Collection

### Option A — tmux launcher (recommended)
Starts C++ deploy, PICO teleop, data exporter, camera viewer, and ZED→PICO bridge in one tmux session:
```bash
python gear_sonic/scripts/launch_data_collection.py --task-prompt "pick up the cup" --with-dex1-grippers --dex1-network-interface wlp0s20f3 --camera-host 192.168.123.164 --with-head
```
- Add `--no-camera-viewer` to skip the camera viewer pane.
- Add `--no-zed-pico-bridge` to skip the ZED→PICO video bridge pane.
- `--camera-host` is used by all panes (data exporter, camera viewer, ZED→PICO bridge).

### Option B — manual (separate terminals)

**Terminal 1 — C++ deploy:**
```bash
cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager real
```

**Terminal 2 — PICO teleop:**
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --vis_vr3pt \
    --dex1_interface wlp0s20f3
```

**Terminal 3 — Data exporter:**
```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --with-dex1-grippers \
    --dex1-network-interface wlp0s20f3 \
    --camera-host 192.168.123.164 \
    --with-head \
    --head-zmq-host 192.168.123.164
```

**Terminal 4 — ZED→PICO bridge:**
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/zed_pico_zmq.py --zmq-host 192.168.123.164
```

### Recording Controls

#### PICO VR controllers

| Input | Action |
|-------|--------|
| Left Grip + A | Toggle recording — starts or stops and saves the current episode |
| Left Grip + B | Discard the current episode without saving |

#### Gripper control

| Input | Action |
|-------|--------|
| Left index trigger | Left gripper open/close |
| Right index trigger | Right gripper open/close |
| Hold left menu button | Suppresses gripper commands (for repositioning) |

---

## Inference

### GPU server (lamb)
Start the GR00T policy server with the finetuned checkpoint:
```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path examples/g1_real_finetune_out/checkpoint-20000 \
    --embodiment-tag UNITREE_G1_SONIC_DEX1 \
    --device cuda:0 \
    --port 5550
```

### Laptop — Option A: tmux launcher (recommended)
```bash
python gear_sonic/scripts/launch_inference.py \
    --policy-host <lamb_ip> \
    --policy-port 5550 \
    --embodiment-tag unitree_g1_sonic_dex1 \
    --with-dex1-grippers \
    --dex1-network-interface wlp0s20f3 \
    --camera-host 192.168.123.164 \
    --with-head \
    --prompt "pick up the cup"
```
Add `--no-data-exporter` to skip the recording pane.

### Laptop — Option B: manual (separate terminals)

**Terminal 1 — C++ deploy:**
```bash
cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager real
```

**Terminal 2 — Keyboard publisher:**
```bash
source .venv_inference/bin/activate
python -c "import zmq,time; ctx=zmq.Context(); pub=ctx.socket(zmq.PUB); pub.bind('tcp://localhost:5580'); time.sleep(0.5); print('Ready — type k, i, p etc'); [pub.send_string(input()) or print('Sent') for _ in iter(int,1)]"
```

**Terminal 3 — VLA inference:**
```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/run_vla_inference.py \
    --host <lamb_ip> \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic_dex1 \
    --with-dex1-grippers \
    --dex1-network-interface wlp0s20f3 \
    --camera-host 192.168.123.164 \
    --with-head \
    --head-zmq-host 192.168.123.164 \
    --prompt "pick up the cup"
```

### Inference keyboard controls (type in keyboard publisher terminal)

| Key | Action |
|-----|--------|
| `k` | Start / stop C++ control loop |
| `i` | Send initial pose and switch to POSE mode |
| `p` | Pause / resume policy loop |
| `[` | Toggle left hand open/closed (initial pose) |
| `]` | Toggle right hand open/closed (initial pose) |
| `t <text>` | Change inference prompt at runtime |

---

## Network / Ports

| Port | Protocol | Publisher | Subscribers | Purpose |
|------|----------|-----------|-------------|---------|
| 5550 | ZMQ REQ/REP | GPU server (lamb) | Laptop inference script | GR00T PolicyServer — inference requests/responses |
| 5555 | ZMQ PUB | Robot (192.168.123.164) ZMQ bridge | Laptop data exporter, inference script | Camera frames (`ego_view` + `stereo_view`) |
| 5556 | ZMQ PUB | Laptop pico_manager / C++ deploy | Data exporter, C++ deploy | Action commands to C++ deploy loop (motion tokens, hand joints); also SMPL pose to data exporter |
| 5557 | ZMQ PUB | Laptop C++ deploy | Inference script, data exporter | Robot state out of C++ deploy (joint pos/vel, projected gravity) |
| 5558 | ZMQ PUB | Robot `bridge_node` | Laptop data exporter / inference | Actual head joint state (`head_state` topic) — also reused as policy server port in MuJoCo sim (no conflict since head and sim are mutually exclusive) |
| 5580 | ZMQ PUB | Laptop keyboard publisher | C++ deploy | Keyboard keypresses — `k` starts the control loop |
| 13579 | TCP | Laptop XRoboToolkit service | PICO | XRoboToolkit command channel |
| 12345 | TCP | Laptop `zed_pico_zmq.py` | PICO | H.264 video stream to PICO headset |

---

## See Also

- `docs/dex1_gripper_integration.md` — Dex1-1 gripper architecture and dataset schema
- `headControl/src/headctrl/head_zmq_bridge/README.md` — ZMQ camera bridge details
