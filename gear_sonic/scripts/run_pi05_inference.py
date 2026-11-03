"""
pi0.5 inference runner for the Sonic whole-body control stack (openpi websocket server).

Same loop, sensors, keyboard keys and real-time chunking as run_vla_inference.py (it runs that script's ``main``);
only the policy differs: an openpi server (``scripts/serve_policy_rtc.py``) with a pi0.5 checkpoint trained on the
gear_sonic exporter data, e.g. config ``pi05_g1_sonic_cube`` (kjswaroopNU/g1-cube-wholebody). Its 68-D action rows
[motion token 64, left gripper, right gripper, head pitch / yaw] are the GR00T unitree_g1_sonic_dex1_3cam actions.

The openpi server is stateless, so RTC lives here: the adapter keeps the last chunk and sends its rows from
``rtc_executed_steps`` on as ``rtc_prev_actions``, with ``rtc_frozen_steps`` as ``rtc_inference_delay``.

  # lamb: cd ~/openpi && .venv/bin/python scripts/serve_policy_rtc.py --policy.config pi05_g1_sonic_cube \\
  #           --policy.dir /data/users/vwh1509/pi05_g1_sonic_cube/30000 --port 8000
  python gear_sonic/scripts/run_pi05_inference.py --host lamb.mech.northwestern.edu --port 8000 \\
      --with-dex1-grippers --dex1-network-interface enp0s31f6 --with-head --head-zmq-host 192.168.123.164 \\
      --camera-host 192.168.123.164 --prompt "Pick up the cube"

Needs openpi-client in this venv (uv pip install -e <openpi>/packages/openpi-client).
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vla_inference  # noqa: E402

TOKEN_DIM = 64
STATE_KEYS = ("left_gripper_state", "right_gripper_state", "projected_gravity", "head_state")
CAMERAS = ("ego_view", "left_wrist", "right_wrist")


@dataclass
class Pi05InferenceConfig(run_vla_inference.InferenceConfig):
    """run_vla_inference.py's options, with the openpi server's defaults."""

    port: int = 8000
    """The port of the openpi websocket server."""

    embodiment_tag: str = "unitree_g1_sonic_dex1_3cam"
    """Only printed (the openpi config fixes the inputs and actions)."""

    prompt: str = "Pick up the cube"
    """The language prompt (the training task)."""

    policy_fps: float = 30.0
    """pi05_g1_sonic_cube_30fps rows are 30 fps (50 rows = 1.67 s); use 50 for the older 50 fps pi05_g1_sonic_cube."""

    execute_steps: int = 20
    """Chunk rows executed per chunk (the HF recipe's RTC execution horizon at 30 fps)."""


def to_openpi_observation(observation: dict) -> dict:
    """run_vla_inference.py's observation -> the openpi g1_sonic_policy inputs (the dataset keys).

    observation/state is the 43-D whole-body q the data exporter recorded as observation.state. Needs
    --with-dex1-grippers and --with-head (gripper and head state are model inputs).
    """
    state = observation["state"]
    missing = [key for key in STATE_KEYS if key not in state]
    assert not missing, f"missing {missing}: run with --with-dex1-grippers and --with-head"
    return {
        "observation/state": np.asarray(observation["q"], dtype=np.float32).reshape(-1),
        **{f"observation/{key}": np.asarray(state[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS},
        **{f"observation/{key}": observation["video"][key][0, 0] for key in CAMERAS},
        "prompt": observation["language"]["annotation.human.task_description"][0][0],
    }


class OpenpiPolicy:
    """An openpi websocket policy behind the PolicyClient calls run_vla_inference.py makes (ping, get_action)."""

    def __init__(self, host: str, port: int):
        from openpi_client import websocket_client_policy

        print(f"Connecting to openpi server at {host}:{port} (waits until it is up)...")
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._previous = None  # the last chunk (T, 68), for RTC

    def ping(self) -> bool:
        return True  # the constructor only returns once connected

    def get_action(self, observation: dict, options: dict | None = None):
        options = options or {}
        if options.get("rtc_reset"):
            self._previous = None
        request = dict(observation)
        if "rtc_executed_steps" in options and self._previous is not None:
            rest = self._previous[int(options["rtc_executed_steps"]):]
            if len(rest):
                request["rtc_prev_actions"] = rest
                request["rtc_inference_delay"] = int(options.get("rtc_frozen_steps", 0))
        actions = np.asarray(self._client.infer(request)["actions"], dtype=np.float32)
        self._previous = actions
        rows = actions[np.newaxis]  # (1, T, 68)
        action = {
            "motion_token": rows[..., :TOKEN_DIM],
            "left_gripper_cmd": rows[..., TOKEN_DIM:TOKEN_DIM + 1],
            "right_gripper_cmd": rows[..., TOKEN_DIM + 1:TOKEN_DIM + 2],
            "head_cmd": rows[..., TOKEN_DIM + 2:TOKEN_DIM + 4],
        }
        return action, {}


if __name__ == "__main__":
    config = tyro.cli(Pi05InferenceConfig)
    assert config.with_dex1_grippers and config.with_head, "pi0.5 needs --with-dex1-grippers and --with-head"
    run_vla_inference.main(
        config, policy=OpenpiPolicy(config.host, config.port), observation_transform=to_openpi_observation
    )
