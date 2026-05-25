# Run Guide

Full pipelines for teleoperated data collection and VLA inference with ZED stereo camera and Dex1-1 grippers.

## Prerequisites

- PICO VR headset paired in XRoboToolkit app
- `video_source.yml` pushed to PICO:
  ```bash
  adb push ~/video_source.yml /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
  ```
- Robot network interface name known (find with `ip link show`, e.g. `wlp0s20f3`)
- Robot Jetson NX clock synced to avoid image latency:
  ```bash
  sudo ntpdate pool.ntp.org   # run on robot (192.168.123.164)
  ```

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
    --with-head
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
| 5558 | ZMQ REQ/REP | Sim only | Sim inference client | Policy server in MuJoCo sim (avoids conflict with 5555) |
| 5580 | ZMQ PUB | Laptop keyboard publisher | C++ deploy | Keyboard keypresses — `k` starts the control loop |
| 13579 | TCP | Laptop XRoboToolkit service | PICO | XRoboToolkit command channel |
| 12345 | TCP | Laptop `zed_pico_zmq.py` | PICO | H.264 video stream to PICO headset |

---

## See Also

- `docs/dex1_gripper_integration.md` — Dex1-1 gripper architecture and dataset schema
- `headControl/src/headctrl/head_zmq_bridge/README.md` — ZMQ camera bridge details
