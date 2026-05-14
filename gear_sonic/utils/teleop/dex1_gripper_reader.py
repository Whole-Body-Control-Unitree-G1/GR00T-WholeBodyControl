import time

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


class Dex1GripperReader:
    """DDS subscriber for a single Dex1-1 parallel gripper motor state.

    Reads from the dex_1_1_service running on PC2.
    Call ChannelFactoryInitialize before instantiating.
    """

    def __init__(self, is_left: bool):
        self.is_left = is_left
        topic = "rt/dex1/left/state" if is_left else "rt/dex1/right/state"
        self._sub = ChannelSubscriber(topic, MotorStates_)
        self._sub.Init(None, 0)

    def read(self) -> dict | None:
        """Return latest gripper state or None if no message available."""
        state = self._sub.Read()
        if state is None:
            return None
        s = state.states[0]
        return {
            "q": float(s.q),
            "dq": float(s.dq),
            "tau_est": float(s.tau_est),
            "receive_timestamp": time.time(),
        }
