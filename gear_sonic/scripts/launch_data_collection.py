"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in a single tmux session:

    Window 0 — data_collection (5 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    ├───────────────────────┼───────────────────────┤
    │ Pane 1: Teleop        │ Pane 3: Camera Viewer │
    │ (.venv_teleop)        │ (.venv_data_collection)│
    │                       ├───────────────────────┤
    │                       │ Pane 4: ZED→PICO      │
    │                       │ (.venv_teleop)        │
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

    Window 2 — robot  (only with --with-robot / --with-control-page, tiled):
        Orin over ssh: Dex1 gripper server, cameras.launch.py (ZED Mini + 2x D405), head_zmq_bridge,
        head_control_node, head_pico_bridge; laptop: control page (decoupled_wbc container, :8080)

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_pico.sh            -> .venv_teleop
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_data_collection.py                          # real robot (default)
    python gear_sonic/scripts/launch_data_collection.py --sim                    # MuJoCo sim
    python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer       # skip viewer
    python gear_sonic/scripts/launch_data_collection.py --pico-input-source isaac-teleop  # in-process CloudXR / DeviceIO
    python gear_sonic/scripts/launch_data_collection.py --with-robot --with-control-page \\
        --with-dex1-grippers --dex1-network-interface enp0s31f6 --with-head \\
        --camera-host 192.168.123.164 --record-wrist-cameras          # robot + laptop in one go
    python gear_sonic/scripts/launch_data_collection.py --stop       # stop everything (or Ctrl+\\ in tmux)
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import signal
import socket
import subprocess
import sys
import time


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    deploy_motor_kp_scale: str = ""
    """Kp scale specification for hardware motor indices (for example, 4,10=1.5)."""

    deploy_motor_kd_scale: str = ""
    """Kd scale specification for hardware motor indices (for example, 4,10=1.5)."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    prompt_file: str = ""
    """Path to a JSON file with a list of task prompts to cycle through with left grip + X
    (only while IDLE). Empty = use --task-prompt only."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback via espeak (data exporter)."""

    # Dex1-1 grippers
    with_dex1_grippers: bool = False
    """Record Dex1-1 parallel gripper state and commands."""

    dex1_network_interface: str = ""
    """Local network interface for Dex1-1 DDS (e.g. wlp0s20f3). Empty = auto."""

    # 2-DOF head
    with_head: bool = False
    """Record 2-DOF head joint state and command via ZMQ bridge_node."""

    # Rerun visualizer
    with_rerun: bool = False
    """Launch Rerun visualizer window during data collection."""

    # ZED → PICO video bridge
    zed_pico_bridge: bool = True
    """Run zed_pico_zmq.py to stream ZED stereo view to PICO headset."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host (shared by data exporter and viewer)."""

    camera_port: int = 5555
    """Camera server port (shared by data exporter and viewer)."""

    zed_pico_fps: int = 30
    """ZED stereo rate on the Orin; the ZED→PICO encoder spreads its bitrate over this many frames per second."""

    # Robot (Orin) side
    with_robot: bool = False
    """Also start the Orin side over ssh in a 'robot' window: Dex1 gripper server, cameras (ZED Mini + 2x D405),
    head_zmq_bridge (synced ego_view / left_wrist / right_wrist on :5555, stereo_view on :5559), head_control_node
    and head_pico_bridge."""

    robot_ssh: str = "unitree@192.168.123.164"
    """ssh target of the Orin (key-based login)."""

    laptop_interface: str = "enp0s31f6"
    """Laptop port on the robot network; head_pico_bridge reads the PICO pose (:5556) from its IP."""

    orin_interface: str = "enP8p1s0"
    """Orin port on the robot network (gripper server and head ROS 2 traffic)."""

    # Control page (decoupled_wbc/scripts/robot_web.py)
    with_control_page: bool = False
    """Start the browser control page (http://localhost:8080) in the decoupled_wbc container: camera previews,
    head / gripper state and control, body status. Don't drive head or grippers from it while teleop is on."""

    control_page_container: str = "g1_wbc"
    """Docker container for the control page (started from control_page_image if not running)."""

    control_page_image: str = "nvgear/gr00t_wbc:latest"
    """Image with ROS 2 Humble + unitree_sdk2py."""

    head_control_dir: str = "~/wbcG1/headControl"
    """Laptop copy of headControl (the control page reads the head calibration from it)."""

    stop: bool = False
    """Stop a running session: Ctrl+C in every pane, wait for the Orin processes to exit, close the session."""


SESSION_NAME = "sonic_data_collection"

# Orin side
ORIN_PROCS = (
    "[d]ex1_1_gripper_server|[h]ead_control_node|[h]ead_pico_bridge|[r]ealsense2_camera_node|[z]ed_wrapper|"
    "[h]ead_zmq_bridge|[c]ameras.launch.py"
)  # [x]yz patterns so the ssh command line itself never matches
U2D2 = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNC7-if00-port0"
ROS_ORIN = (
    "source /opt/ros/humble/setup.bash && source ~/wbcG1/headControl/install/setup.bash && "
    "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
)
# Camera drivers and the camera bridge talk over loopback; the head nodes over the robot ethernet, so the laptop's
# control page sees head/state too.
CDDS_LOOPBACK = "CYCLONEDDS_URI=file://$HOME/robot_cameras/cyclone_localhost.xml"
CDDS_ETH = "CYCLONEDDS_URI=file://$HOME/wbcG1/cyclone_eth.xml"


def _cdds_xml(interface: str) -> str:
    """CycloneDDS config using only one network interface."""
    return (
        '<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="'
        f'{interface}"/></Interfaces></General></Domain></CycloneDDS>\n'
    )


def _orin(config: DataCollectionLaunchConfig, cmd: str, stdin: str | None = None):
    """Run a command on the Orin (non-interactive ssh)."""
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", config.robot_ssh, cmd],
        input=stdin, capture_output=True, text=True,
    )


def _interface_ip(interface: str) -> str | None:
    result = subprocess.run(["ip", "-4", "-o", "addr", "show", interface], capture_output=True, text=True)
    fields = result.stdout.split()
    return fields[fields.index("inet") + 1].split("/")[0] if "inet" in fields else None


def _check_prerequisites(config: DataCollectionLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
        errors.append(
            ".venv_data_collection not found. Run: "
            "bash install_scripts/install_data_collection.sh"
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(see install instructions)."
        )

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")

    if config.with_robot:
        if not _interface_ip(config.laptop_interface):
            errors.append(
                f"{config.laptop_interface} has no IPv4 address: plug in the robot ethernet "
                "(or set --laptop-interface)"
            )
        if _orin(config, "true").returncode != 0:
            errors.append(f"cannot ssh to {config.robot_ssh} (robot on? key-based login set up?)")
        else:
            running = _orin(config, f"ps -eo pid,args | grep -E '{ORIN_PROCS}' | cut -c1-100").stdout.strip()
            if running:
                errors.append(
                    "already running on the Orin (stop those first, or "
                    f"'python {Path(__file__).name} --stop'):\n{running}"
                )
            if _orin(config, f"test -w {U2D2}").returncode != 0:
                errors.append(f"head U2D2 missing or not writable on the Orin ({U2D2})")
            grip_ports = _orin(config, "ls /dev/serial/by-id/ | grep -c FTBRJ9EO").stdout.strip()
            if int(grip_ports or 0) < 4:
                print("WARNING: gripper serial hub (FTBRJ9EO) not found on the Orin: grippers will not start")

    if config.with_control_page and not shutil.which("docker"):
        errors.append("--with-control-page needs docker")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session():
    """Create a 4-pane tmux layout."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "data_collection"],
    )

    # Split into 4 panes:
    #   0 | 1
    #   -----
    #   2 | 3

    # Split horizontally: pane 0 (left) and pane 1 (right)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )

    # Split left pane vertically: pane 0 (top-left) and pane 2 (bottom-left)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )

    # Split right pane vertically: pane 1 becomes top-right, new pane 3 bottom-right
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    # Split bottom-right pane again: pane 3 becomes camera viewer, pane 4 is ZED→PICO bridge
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.3", "-v"],
    )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _create_robot_window(n_panes: int):
    """A 'robot' window with n_panes tiled panes (Orin processes, control page)."""
    subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, "-n", "robot"])
    for _ in range(n_panes - 1):
        subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:robot"])
        # re-tile after every split, else the detached session runs out of room for new panes
        subprocess.run(["tmux", "select-layout", "-t", f"{SESSION_NAME}:robot", "tiled"])
    time.sleep(5)


def _send_to_robot_pane(pane_index: int, cmd: str, wait: float = 0.5):
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:robot.{pane_index}", cmd, "C-m"])
    time.sleep(wait)


def _start_robot(config: DataCollectionLaunchConfig):
    """Orin side in the 'robot' window, panes 0-4 (the same processes as decoupled_wbc's launch_g1.sh)."""
    _orin(config, "cat > ~/wbcG1/cyclone_eth.xml", stdin=_cdds_xml(config.orin_interface))
    ssh_t = f"ssh -t -o ConnectTimeout=5 -o ServerAliveInterval=5 {config.robot_ssh}"
    laptop_ip = _interface_ip(config.laptop_interface)

    print("Starting Dex1 gripper server (robot pane 0, type the sudo password there)...")
    _send_to_robot_pane(
        0,
        f"{ssh_t} 'cd ~/dex1_1_service/bin && sudo ./dex1_1_gripper_server --network {config.orin_interface}'",
    )

    print("Starting cameras: ZED Mini + 2x D405 (robot pane 1)...")
    _send_to_robot_pane(
        1,
        f"{ssh_t} 'source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && "
        f"export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp {CDDS_LOOPBACK} && "
        "exec ros2 launch ~/robot_cameras/cameras.launch.py'",
    )

    # the bridge subscribes on loopback like the camera drivers; give them time to come up
    print("Starting camera bridge (robot pane 2)...")
    _send_to_robot_pane(
        2,
        f"{ssh_t} '{ROS_ORIN} && export {CDDS_LOOPBACK} && sleep 10 && exec ros2 run head_zmq_bridge "
        "head_zmq_bridge --ros-args --params-file "
        "~/wbcG1/headControl/install/head_zmq_bridge/share/head_zmq_bridge/config/camera.yaml'",
    )

    # The head starts only after the gripper server has found its motors: that probe opens every /dev/ttyUSB*, the
    # head's U2D2 included, and re-configures the port under a head node that is already running. Up to 2 min, so
    # there is time to type the sudo password.
    wait_grippers = (
        "echo 'waiting for the gripper server (robot pane 0) to finish probing the serial ports...'; "
        "for i in $(seq 120); do "
        f"tmux capture-pane -p -S -200 -t {SESSION_NAME}:robot.0 | grep -q 'Gripper Server started' && break; "
        "sleep 1; done"
    )
    print("Starting head_control_node after the gripper server (robot pane 3)...")
    _send_to_robot_pane(
        3,
        f"{wait_grippers}; {ssh_t} '{ROS_ORIN} && export {CDDS_ETH} && "
        "exec ros2 launch head_control head_control.launch.xml'",
    )

    print(f"Starting head_pico_bridge, PICO pose from {laptop_ip}:5556 (robot pane 4)...")
    _send_to_robot_pane(
        4,
        f"{wait_grippers}; {ssh_t} '{ROS_ORIN} && export {CDDS_ETH} && "
        f"exec ros2 run head_pico_bridge head_pico_bridge --ros-args -p zmq_host:={laptop_ip}'",
    )


def _start_control_page(config: DataCollectionLaunchConfig, repo_root: Path, pane_index: int):
    """robot_web.py in the decoupled_wbc container (ROS 2 + unitree_sdk2py are not in the gear_sonic venvs)."""
    container = config.control_page_container
    head_control_dir = Path(config.head_control_dir).expanduser()
    running = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container], capture_output=True, text=True
    ).stdout.strip()
    if running != "true":
        print(f"Starting container {container} ({config.control_page_image})...")
        # --ulimit core=0: a crashing process would otherwise write a ~1 GB root-owned "core" into the repo (its cwd)
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", container, "--network", "host", "--ulimit", "core=0",
             "-e", f"PYTHONPATH={repo_root}", "-v", f"{repo_root}:{repo_root}",
             "-v", f"{head_control_dir}:{head_control_dir}:ro", "-w", str(repo_root),
             config.control_page_image, "sleep", "infinity"],
            check=True, capture_output=True,
        )
    subprocess.run(
        ["docker", "exec", "-i", container, "sh", "-c", "cat > /tmp/cyclone_eth.xml"],
        input=_cdds_xml(config.laptop_interface), text=True,
    )
    calib = head_control_dir / "src/headctrl/head_control/config/head_joint_calib.yaml"
    print(f"Starting control page http://localhost:8080 (robot pane {pane_index})...")
    _send_to_robot_pane(
        pane_index,
        f"docker exec -it {container} bash -c 'source /opt/ros/humble/setup.bash && source /root/venv/bin/activate && "
        "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///tmp/cyclone_eth.xml "
        "PYTHONPATH=$PWD:$PYTHONPATH && exec python decoupled_wbc/scripts/robot_web.py "
        f"--interface {config.laptop_interface} --head-calib {calib}'",
    )


def _stop_session(config: DataCollectionLaunchConfig):
    """Ctrl+C in every pane, wait for the Orin processes to exit, then close the session."""
    if subprocess.run(["tmux", "has-session", "-t", SESSION_NAME], capture_output=True).returncode == 0:
        panes = subprocess.run(
            ["tmux", "list-panes", "-s", "-t", SESSION_NAME, "-F", "#{session_name}:#{window_index}.#{pane_index}"],
            capture_output=True, text=True,
        ).stdout.split()
        print(f"Ctrl+C in {len(panes)} panes...")
        for pane in panes:
            subprocess.run(["tmux", "send-keys", "-t", pane, "C-c"])
        for _ in range(15):
            if not _orin(config, f"ps -eo args | grep -E '{ORIN_PROCS}'").stdout.strip():
                break
            time.sleep(1)
        subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)
    else:
        print(f"No tmux session '{SESSION_NAME}'")
    left = _orin(config, f"ps -eo pid,args | grep -E '{ORIN_PROCS}' | cut -c1-100")
    if left.stdout.strip():
        print(f"Still running on the Orin (kill by hand):\n{left.stdout}")
    elif left.returncode != 255:  # 255 = ssh failed; grep's 1 = nothing left
        print("Orin: all stopped")


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    if config.stop:
        _stop_session(config)
        return

    _check_prerequisites(config)
    _kill_existing_session()

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Teleop input:    {config.pico_input_source}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  DC frequency:    {config.data_exporter_frequency} Hz")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Text-to-speech:  {'Yes' if config.text_to_speech else 'No'}")
    print(f"  PC IP (for PICO): {_get_local_ip()}")
    print(f"  Teleop vis:      vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print("=" * 60)

    _create_tmux_session()
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window 'robot' (--with-robot / --with-control-page): Orin side first, so the cameras warm up ---
    if config.with_robot or config.with_control_page:
        n_robot_panes = (5 if config.with_robot else 0) + (1 if config.with_control_page else 0)
        _create_robot_window(n_robot_panes)
        if config.with_robot:
            _start_robot(config)
            # Ctrl+\ stops the Orin processes cleanly (Ctrl+C everywhere) instead of only dropping the ssh sessions
            subprocess.run(
                ["tmux", "bind-key", "-T", "root", "C-\\", "run-shell", "-b",
                 f"{sys.executable} {Path(__file__).resolve()} --stop --robot-ssh {config.robot_ssh}"],
            )
        if config.with_control_page:
            _start_control_page(config, repo_root, n_robot_panes - 1)
        subprocess.run(["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"])

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        # Switch back to the data_collection window for the remaining panes
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    if config.deploy_motor_kp_scale:
        deploy_cmd += f"--motor-kp-scale {config.deploy_motor_kp_scale} "
    if config.deploy_motor_kd_scale:
        deploy_cmd += f"--motor-kd-scale {config.deploy_motor_kd_scale} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): Teleop Streamer ---
    pico_cmd = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source}"
    )
    if config.pico_manager:
        pico_cmd += " --manager"
    if config.pico_vis_vr3pt:
        pico_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_cmd += " --waist_tracking"
    if config.with_dex1_grippers and config.dex1_network_interface:
        pico_cmd += f" --dex1_interface {config.dex1_network_interface}"

    print("Starting teleop streamer (pane 2)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Camera Viewer ---
    if config.camera_viewer:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        print("Starting camera viewer (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)

    # --- Pane 4 (bottom-right split): ZED → PICO video bridge ---
    if config.zed_pico_bridge:
        zed_cmd = (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python gear_sonic/scripts/zed_pico_zmq.py --zmq-host {config.camera_host} "
            f"--fps {config.zed_pico_fps}"
        )
        print("Starting ZED→PICO bridge (pane 4)...")
        _send_to_pane(4, zed_cmd, wait=1.0)

    # --- Pane 1 (top-right): Data Exporter ---
    exporter_cmd = (
        f"cd {repo_root} && "
        f"source .venv_data_collection/bin/activate && "
        f"python gear_sonic/scripts/run_data_exporter.py "
        f"--task-prompt '{config.task_prompt}' "
        f"--data-collection-frequency {config.data_exporter_frequency} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port}"
    )
    if config.prompt_file:
        exporter_cmd += f" --prompt-file '{config.prompt_file}'"
    if config.dataset_name:
        exporter_cmd += f" --dataset-name '{config.dataset_name}'"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"
    if config.with_dex1_grippers:
        exporter_cmd += f" --with-dex1-grippers --dex1-network-interface {config.dex1_network_interface}"
    if config.with_head:
        exporter_cmd += f" --with-head --head-zmq-host {config.camera_host}"
    if config.with_rerun:
        exporter_cmd += " --with-rerun"

    print("Starting data exporter (pane 1)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Select the data exporter pane so the user lands there for interactive input
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    print("  Window 'data_collection':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Teleop Streamer")
    print("    Pane 2 (top-right):    Data Exporter  <-- you are here")
    if config.camera_viewer:
        print("    Pane 3 (bottom-right): Camera Viewer")
    if config.with_robot or config.with_control_page:
        print()
        print("  Window 'robot':")
        if config.with_robot:
            print("    Pane 0: Dex1 gripper server   <-- type the Orin sudo password here first")
            print("    Pane 1: cameras (ZED Mini + 2x D405)")
            print("    Pane 2: head_zmq_bridge (:5555 synced cameras, :5559 stereo)")
            print("    Pane 3: head_control_node (starts after the gripper server)")
            print("    Pane 4: head_pico_bridge")
        if config.with_control_page:
            print("    Last pane: control page -> http://localhost:8080")
    print()
    if config.with_robot:
        print("  ** robot window, pane 0: type the sudo password for the gripper server **")
    print("  ** deploy.sh (pane 0) is waiting for confirmation —")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    if config.with_robot or config.with_control_page:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session" + (" (stops the Orin side first)" if config.with_robot else ""))
    print("=" * 60)

    # Attach to the session
    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DataCollectionLaunchConfig)
    main(config)
