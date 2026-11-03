"""
All-in-one tmux launcher for pi0.5 inference on the real G1: launch_inference.py with run_pi05_inference.py as the
client and the pi0.5 defaults (same tmux session, panes, keyboard keys and --stop behaviour).

Starts the Orin side (--with-robot: Dex1 gripper server, cameras, camera bridge, head_control_node,
head_pico_bridge), the C++ SONIC deploy, the keyboard publisher, the data exporter and the pi0.5 client, which
talks to an openpi server with a pi0.5 SONIC checkpoint, e.g. on lamb:

    cd ~/openpi && CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_policy_rtc.py \\
        --policy.config pi05_g1_sonic_cube --policy.dir /data/users/vwh1509/pi05_g1_sonic_cube/30000 --port 8000

Usage (from repo root):
    python gear_sonic/scripts/launch_pi05_inference.py
    python gear_sonic/scripts/launch_pi05_inference.py --prompt "Pick up the cube" --no-data-exporter

Then: Enter in the deploy pane (Proceed with deployment?), sudo password in the gripper pane (window 'robot'),
and in the keyboard pane k (SONIC on), i (initial pose), p (policy on / off).
"""

from dataclasses import dataclass
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_inference  # noqa: E402  (re-execs with .venv_inference if tyro is missing)

import tyro  # noqa: E402


@dataclass
class Pi05InferenceLaunchConfig(launch_inference.InferenceLaunchConfig):
    """launch_inference.py's options with the pi0.5 defaults."""

    inference_script: str = "gear_sonic/scripts/run_pi05_inference.py"
    """Inference client (the pi0.5 / openpi one)."""

    policy_host: str = "lamb.mech.northwestern.edu"
    """openpi server host."""

    policy_port: int = 8000
    """openpi server port."""

    embodiment_tag: str = "unitree_g1_sonic_dex1_3cam"
    """Only printed (the openpi config fixes the inputs and actions)."""

    prompt: str = "Pick up the cube"
    """Language prompt (the training task)."""

    camera_host: str = "192.168.123.164"
    """Camera server host (head_zmq_bridge on the Orin)."""

    with_robot: bool = True
    """Start the Orin processes too."""

    with_dex1_grippers: bool = True
    """Dex1-1 grippers (their state is a pi0.5 input)."""

    dex1_network_interface: str = "enp0s31f6"
    """Laptop port on the robot network, for Dex1 DDS."""

    with_head: bool = True
    """2-DOF head (its state is a pi0.5 input)."""

    execute_steps: int = 20
    """Chunk rows executed per chunk (rows are 30 fps for pi05_g1_sonic_cube_30fps)."""


if __name__ == "__main__":
    signal.signal(signal.SIGINT, launch_inference._signal_handler)  # noqa: SLF001
    launch_inference.main(tyro.cli(Pi05InferenceLaunchConfig))
