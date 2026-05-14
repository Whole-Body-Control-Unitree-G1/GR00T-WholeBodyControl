import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_


MAX_GRIPPER_ANGLE = 5.5  # rad, from dex1_1_service test script


class Dex1GripperSender:
    """DDS publisher for a single Dex1-1 parallel gripper motor.

    Publishes position commands to the dex_1_1_service running on PC2.
    Call ChannelFactoryInitialize before instantiating.
    """

    def __init__(self, is_left: bool):
        self.is_left = is_left
        topic = "rt/dex1/left/cmd" if is_left else "rt/dex1/right/cmd"
        self._pub = ChannelPublisher(topic, MotorCmds_)
        self._pub.Init()
        self._cmd = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_()])
        self._cmd.cmds[0].kp = 5.0
        self._cmd.cmds[0].kd = 0.05

    def send(self, position: float):
        """Send a position command in radians, clamped to [0, MAX_GRIPPER_ANGLE]."""
        self._cmd.cmds[0].q = float(np.clip(position, 0.0, MAX_GRIPPER_ANGLE))
        self._pub.Write(self._cmd)

    def from_trigger(self, trigger: float):
        """Convert a trigger value [0, 1] to a gripper position and send."""
        self.send(trigger * MAX_GRIPPER_ANGLE)
