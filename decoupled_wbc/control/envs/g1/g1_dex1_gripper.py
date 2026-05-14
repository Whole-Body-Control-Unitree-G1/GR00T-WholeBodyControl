import gymnasium as gym
import numpy as np

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.envs.g1.utils.command_sender import Dex1CommandSender
from decoupled_wbc.control.envs.g1.utils.state_processor import Dex1StateProcessor

class G1DexGripper(Env):

    def __init__(self, is_left: bool=True):
        super().__init__()
        self.is_left = is_left
        self.gripper_state_processor = Dex1StateProcessor(is_left=self.is_left)
        self.gripper_command_sender = Dex1CommandSender(is_left=self.is_left)

    
    def observe(self) -> dict[str, any]:
        hand_state = self.gripper_state_processor._prepare_low_state()

        if hand_state is None:
            return {
                "gripper_q": 0.0,
                "gripper_dq": 0.0,
                "gripper_tau_est": 0.0,
            }

        assert hand_state.shape == (1, 3)

        return {
            "gripper_q": hand_state[0, 0],
            "gripper_dq": hand_state[0, 1],
            "gripper_tau_est": hand_state[0, 2],
        }

    def queue_action(self, action: dict[str, any]):
        self.gripper_command_sender.send_command(action["gripper_q"])

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "gripper_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=()),
                "gripper_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=()),
                "gripper_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=()),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {"gripper_q": gym.spaces.Box(low=0.0, high=5.5, shape=())}
        )
