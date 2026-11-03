"""Browser control page for the G1 embodiment beside decoupled_wbc: http://localhost:8080 on the laptop.

Cameras (ZMQ from head_zmq_bridge on the Orin: the time-aligned ego_view / left_wrist / right_wrist JPEG sets the
policy uses), head (ROS 2: head/target, head/state from head_control_node), Dex1 grippers (Unitree DDS:
rt/dex1/{left,right}/cmd|state from dex1_1_gripper_server) and body status (rt/lowstate, rt/lowcmd: IMU, motor
temperatures, whether a control loop is commanding). The body itself is driven by run_g1_control_loop.py.

Head angles are MagicSim's joint angles (joint_yaw +left, joint_pitch +down) in degrees; ranges, the straight-ahead /
level reference and home come from the head calibration yaml. Grippers are shown as opening in % (Dex1 angle 5.5 rad
= fully open, 0 = closed).

Run in the decoupled_wbc container (ROS 2 Humble + unitree_sdk2py) with the robot on ethernet:

  source /opt/ros/humble/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  python decoupled_wbc/scripts/robot_web.py --interface enp0s31f6 \\
      --head-calib ~/wbcG1/headControl/src/headctrl/head_control/config/head_joint_calib.yaml

Don't drive the grippers from here while run_g1_control_loop.py runs with --with-dex1-grippers: both would publish
rt/dex1/*/cmd.
"""

from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import msgpack
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tyro
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
import yaml
import zmq

GRIPPER_OPEN = 5.5  # Dex1 motor angle [rad] when fully open; 0 = closed
GRIPPER_KP, GRIPPER_KD = 5.0, 0.2
CAMERAS = ("ego_view", "left_wrist", "right_wrist")
STREAM_FPS = 15  # browser preview rate per camera (the policy reads the full-rate ZMQ stream itself)


@dataclass
class Args:
    interface: str = "enp0s31f6"
    """Laptop network interface on the robot's 192.168.123.x network (Unitree DDS)."""
    head_calib: Path = Path("~/wbcG1/headControl/src/headctrl/head_control/config/head_joint_calib.yaml")
    """head_control's joint calibration (ranges, straight-ahead / level reference, home)."""
    camera_url: str = "tcp://192.168.123.164:5555"
    """head_zmq_bridge's time-aligned camera sets."""
    port: int = 8080


def load_head_calib(path: Path) -> dict:
    doc = yaml.safe_load(path.expanduser().read_text())
    params = next(v["ros__parameters"] for v in doc.values() if "ros__parameters" in v)
    return {
        name: {k: math.degrees(params[name][k]) for k in ("min_angle", "max_angle", "home_angle", "ref_angle")}
        for name in ("joint_yaw", "joint_pitch")
    }


class Robot(Node):
    """Latest camera / head / gripper / body state, and the head / gripper command publishers."""

    def __init__(self, camera_url: str):
        super().__init__("robot_web")
        self.lock = threading.Lock()
        self.head = None  # (yaw deg, pitch deg, monotonic stamp)
        self.grippers = {}  # side -> (MotorStates_, monotonic stamp)
        self.body = None  # (LowState_, monotonic stamp)
        self.lowcmd_stamp = None
        self.frames = {}  # camera -> latest JPEG bytes
        self.frame_id = 0  # increments with every camera set
        self.set_arrivals = deque(maxlen=60)  # monotonic arrival times of camera sets
        self.set_spread = 0.0  # ms between the earliest and latest camera of the last set
        self.new_frame = threading.Condition(self.lock)

        self.head_pub = self.create_publisher(JointState, "head/target", 10)
        self.create_subscription(JointState, "head/state", self.on_head, 10)

        self.gripper_pub = {}
        for side in ("left", "right"):
            pub = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
            pub.Init()
            self.gripper_pub[side] = pub
        # The DDS state topics arrive at ~500 Hz each. Without a handler CycloneDDS keeps the newest samples in C and
        # Python decodes only what poll_dds() takes (a handler would decode every sample: a full CPU core).
        self.dds_lock = threading.Lock()
        self.dds_subs = {}
        for key, topic, msg_type in (("left", "rt/dex1/left/state", MotorStates_),
                                     ("right", "rt/dex1/right/state", MotorStates_),
                                     ("lowstate", "rt/lowstate", LowState_), ("lowcmd", "rt/lowcmd", LowCmd_)):
            sub = ChannelSubscriber(topic, msg_type)
            sub.Init(None, 1)
            self.dds_subs[key] = sub

        threading.Thread(target=self.camera_loop, args=(camera_url,), daemon=True).start()

    def camera_loop(self, url):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.connect(url)
        while True:
            msg = msgpack.unpackb(sock.recv(), raw=False)
            stamps = list(msg["timestamps"].values())
            with self.lock:
                self.frames.update(msg["images"])
                self.frame_id += 1
                self.set_arrivals.append(time.monotonic())
                self.set_spread = 1000.0 * (max(stamps) - min(stamps))
                self.new_frame.notify_all()

    def on_head(self, msg):
        angles = dict(zip(msg.name, msg.position))
        with self.lock:
            self.head = (math.degrees(angles["joint_yaw"]), math.degrees(angles["joint_pitch"]), time.monotonic())

    def poll_dds(self):
        """Take the newest gripper / body samples (called at the page's poll rate)."""
        with self.dds_lock:
            now = time.monotonic()
            for key, sub in self.dds_subs.items():
                msg = sub.Read(timeout=0.002)
                if msg is None:
                    continue
                if key == "lowcmd":
                    self.lowcmd_stamp = now
                elif key == "lowstate":
                    self.body = (msg, now)
                else:
                    self.grippers[key] = (msg, now)

    def send_head(self, yaw_deg, pitch_deg):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint_yaw", "joint_pitch"]
        msg.position = [math.radians(yaw_deg), math.radians(pitch_deg)]
        self.head_pub.publish(msg)

    def send_gripper(self, side, open_pct):
        cmd = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_()])
        cmd.cmds[0].q = GRIPPER_OPEN * min(max(open_pct, 0.0), 100.0) / 100.0
        cmd.cmds[0].kp, cmd.cmds[0].kd = GRIPPER_KP, GRIPPER_KD
        self.gripper_pub[side].Write(cmd)

    def snapshot(self):
        self.poll_dds()
        now = time.monotonic()
        age = lambda stamp: None if stamp is None else now - stamp  # noqa: E731
        grippers, body = dict(self.grippers), self.body
        body_state = None
        if body is not None:
            msg, stamp = body
            roll, pitch, _ = (math.degrees(a) for a in msg.imu_state.rpy)
            temp = max(max(m.temperature) for m in msg.motor_state[:29])
            body_state = {"roll": roll, "pitch": pitch, "temp": temp, "age": age(stamp)}
        with self.lock:
            arrivals = list(self.set_arrivals)
            fps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
            head, spread = self.head, self.set_spread
        return {
            "head": None if head is None else {"yaw": head[0], "pitch": head[1], "age": age(head[2])},
            "grippers": {side: {"open": 100.0 * msg.states[0].q / GRIPPER_OPEN, "q": msg.states[0].q,
                                "age": age(stamp)} for side, (msg, stamp) in grippers.items()},
            "body": body_state,
            "lowcmd_age": age(self.lowcmd_stamp),
            "cameras": {"fps": fps, "spread_ms": spread, "age": age(arrivals[-1]) if arrivals else None},
        }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #070b10; --panel: rgba(14, 20, 28, .82); --panel2: #0f161f; --ink: #d7e3ee; --muted: #6f8295; --line: #1d2a38;
    --accent: #2ee6c9; --accent2: #4da3ff; --accent-dim: rgba(46, 230, 201, .14); --ok: #2ee6a0; --warn: #ffb547; --bad: #ff5d6c;
    --glow: 0 0 0 1px rgba(46, 230, 201, .18), 0 8px 30px rgba(0, 0, 0, .45);
  }
  * { box-sizing: border-box; }
  body { margin: 0; color: var(--ink); font: 14px/1.45 Inter, system-ui, sans-serif; min-height: 100vh; background:
         radial-gradient(900px 500px at 85% -10%, rgba(77, 163, 255, .14), transparent 60%),
         radial-gradient(800px 500px at -10% 110%, rgba(46, 230, 201, .10), transparent 60%),
         linear-gradient(rgba(46, 230, 201, .035) 1px, transparent 1px) 0 0 / 32px 32px,
         linear-gradient(90deg, rgba(46, 230, 201, .035) 1px, transparent 1px) 0 0 / 32px 32px, var(--bg); }
  .wrap { max-width: 1320px; margin: 0 auto; padding: 18px 16px 32px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;
           padding-bottom: 14px; border-bottom: 1px solid var(--line); }
  .brand { display: flex; align-items: baseline; gap: 12px; }
  .brand h1 { font: 700 20px "JetBrains Mono", monospace; margin: 0; letter-spacing: .04em; text-transform: uppercase; }
  .brand h1::before { content: "▍"; color: var(--accent); margin-right: 6px; }
  .brand span { color: var(--muted); font: 500 12px "JetBrains Mono", monospace; letter-spacing: .08em; text-transform: uppercase; }
  .pills { display: flex; flex-wrap: wrap; gap: 8px; }
  .pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 6px; background: var(--panel2);
          border: 1px solid var(--line); font: 500 11.5px "JetBrains Mono", monospace; letter-spacing: .05em; text-transform: uppercase; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.ok { background: var(--ok); box-shadow: 0 0 10px var(--ok); }
  .dot.warn { background: var(--warn); box-shadow: 0 0 10px var(--warn); }
  .dot.bad { background: var(--bad); box-shadow: 0 0 10px var(--bad); animation: blink 1.2s infinite; }
  @keyframes blink { 50% { opacity: .35; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; box-shadow: var(--glow);
          backdrop-filter: blur(6px); position: relative; }
  .card::before { content: ""; position: absolute; left: 0; top: 0; width: 28px; height: 2px; background: var(--accent); border-radius: 2px; }
  .card h2 { font: 600 12px "JetBrains Mono", monospace; letter-spacing: .12em; text-transform: uppercase; margin: 0 0 14px;
             display: flex; align-items: center; justify-content: space-between; color: var(--accent); }
  .card h2 small { color: var(--muted); font-weight: 500; letter-spacing: .04em; text-transform: none; font-family: Inter, sans-serif; }
  .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  .cams { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 14px; }
  .cams .side { display: grid; grid-template-rows: 1fr 1fr; gap: 12px; }
  .cam { position: relative; border-radius: 10px; overflow: hidden; background: #000; aspect-ratio: 4 / 3; border: 1px solid var(--line); box-shadow: var(--glow); }
  .cam img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cam::after { content: ""; position: absolute; inset: 10px; pointer-events: none;
    background: linear-gradient(var(--accent), var(--accent)) 0 0 / 18px 2px, linear-gradient(var(--accent), var(--accent)) 0 0 / 2px 18px,
                linear-gradient(var(--accent), var(--accent)) 100% 0 / 18px 2px, linear-gradient(var(--accent), var(--accent)) 100% 0 / 2px 18px,
                linear-gradient(var(--accent), var(--accent)) 0 100% / 18px 2px, linear-gradient(var(--accent), var(--accent)) 0 100% / 2px 18px,
                linear-gradient(var(--accent), var(--accent)) 100% 100% / 18px 2px, linear-gradient(var(--accent), var(--accent)) 100% 100% / 2px 18px;
    background-repeat: no-repeat; opacity: .7; }
  .cam .tag { position: absolute; left: 16px; top: 16px; padding: 3px 8px; border-radius: 4px; z-index: 1;
              font: 600 11px "JetBrains Mono", monospace; letter-spacing: .08em; text-transform: uppercase;
              background: rgba(7, 11, 16, .72); color: var(--accent); border: 1px solid rgba(46, 230, 201, .35); }
  .cam .hud { position: absolute; right: 16px; bottom: 16px; padding: 3px 8px; border-radius: 4px; font-size: 11px; z-index: 1;
              background: rgba(7, 11, 16, .72); color: var(--ink); border: 1px solid var(--line); }

  .grid { display: grid; grid-template-columns: 1.15fr 1fr .8fr; gap: 14px; }
  .pad { position: relative; border-radius: 8px; border: 1px solid var(--line); aspect-ratio: 16 / 9; cursor: crosshair;
         touch-action: none; overflow: hidden; background:
         linear-gradient(rgba(77, 163, 255, .07) 1px, transparent 1px) 0 0 / 10% 12.5%,
         linear-gradient(90deg, rgba(77, 163, 255, .07) 1px, transparent 1px) 0 0 / 10% 12.5%, #0a1017; }
  .pad .grid-line { position: absolute; background: rgba(46, 230, 201, .35); }
  .pad .lbl { position: absolute; font: 500 10.5px "JetBrains Mono", monospace; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
  .pad .target, .pad .meas { position: absolute; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; }
  .pad .target { width: 20px; height: 20px; border: 2px solid var(--accent); box-shadow: 0 0 14px rgba(46, 230, 201, .6); }
  .pad .meas { width: 8px; height: 8px; background: var(--accent2); box-shadow: 0 0 10px var(--accent2); }
  .sl { display: grid; grid-template-columns: 52px 1fr 112px; gap: 10px; align-items: center; margin: 8px 0; }
  .sl b { font: 600 11px "JetBrains Mono", monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .val { text-align: right; font-size: 12.5px; } .val i { color: var(--accent2); font-style: normal; }
  .btns { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
  button { border: 1px solid var(--line); background: var(--panel2); color: var(--ink); border-radius: 6px; padding: 7px 12px;
           font: 500 12px "JetBrains Mono", monospace; letter-spacing: .04em; cursor: pointer; transition: all .12s; }
  button:hover { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 12px rgba(46, 230, 201, .25); }
  button.primary { background: var(--accent-dim); border-color: rgba(46, 230, 201, .55); color: var(--accent); }
  button.primary:hover { background: var(--accent); color: #04110e; }
  .note { font-size: 12px; color: var(--muted); margin-top: 10px; }
  .status { font: 500 12px "JetBrains Mono", monospace; margin-top: 8px; } .status.ok { color: var(--ok); } .status.bad { color: var(--bad); }

  .grip { padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); margin-bottom: 10px; }
  .grip-top { display: flex; justify-content: space-between; align-items: baseline; }
  .grip-top b { font: 600 11px "JetBrains Mono", monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .big { font-size: 26px; font-weight: 700; color: var(--accent); text-shadow: 0 0 18px rgba(46, 230, 201, .35); }
  .bar { height: 6px; border-radius: 3px; background: #16202b; overflow: hidden; margin: 8px 0 6px; }
  .bar > div { height: 100%; width: 0; background: linear-gradient(90deg, var(--accent2), var(--accent)); box-shadow: 0 0 12px var(--accent); transition: width .15s; }

  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat { padding: 10px 12px; border-radius: 8px; background: var(--panel2); border: 1px solid var(--line); }
  .stat .k { font: 500 10.5px "JetBrains Mono", monospace; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
  .stat .v { font-size: 20px; font-weight: 700; color: var(--ink); }
  .horizon { position: relative; height: 96px; border-radius: 8px; overflow: hidden; margin-top: 10px; border: 1px solid var(--line);
             background: linear-gradient(#0f2a44 50%, #2a1d10 50%); }
  .horizon .line { position: absolute; left: -20%; width: 140%; top: 50%; height: 2px; background: var(--accent2); box-shadow: 0 0 10px var(--accent2); transform-origin: 50% 50%; }
  .horizon .cross { position: absolute; left: 50%; top: 50%; width: 52px; height: 2px; background: var(--warn); transform: translate(-50%, -50%); box-shadow: 0 0 8px var(--warn); }

  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr 1fr; } .grid > :last-child { grid-column: span 2; } }
  @media (max-width: 680px) { .cams, .grid { grid-template-columns: 1fr; } .cams .side { grid-template-rows: none; grid-template-columns: 1fr 1fr; }
                              .grid > :last-child { grid-column: auto; } }
</style></head>
<body><div class="wrap">
  <header>
    <div class="brand"><h1>G1 Control</h1><span>HOMIE · Dex1 · TWIST2 head</span></div>
    <div class="pills">
      <span class="pill"><span class="dot" id="p-body"></span>Robot state</span>
      <span class="pill"><span class="dot" id="p-loop"></span><span id="p-loop-t">Control loop</span></span>
      <span class="pill"><span class="dot" id="p-cams"></span>Cameras <span class="mono" id="p-cams-t"></span></span>
      <span class="pill"><span class="dot" id="p-head"></span>Head</span>
      <span class="pill"><span class="dot" id="p-grip"></span>Grippers</span>
    </div>
  </header>

  <section class="cams">
    <div class="cam"><img src="/stream/ego_view" alt="head camera"><span class="tag">Head · ZED Mini</span><span class="hud mono" id="hud"></span></div>
    <div class="side">
      <div class="cam"><img src="/stream/left_wrist" alt="left wrist camera"><span class="tag">Left wrist · D405</span></div>
      <div class="cam"><img src="/stream/right_wrist" alt="right wrist camera"><span class="tag">Right wrist · D405</span></div>
    </div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Head <small>click or drag the pad to look there</small></h2>
      <div class="pad" id="pad">
        <div class="grid-line" id="gl-v" style="top:0;bottom:0;width:1px"></div>
        <div class="grid-line" id="gl-h" style="left:0;right:0;height:1px"></div>
        <span class="lbl" style="left:8px;top:6px">← left</span><span class="lbl" style="right:8px;top:6px">right →</span>
        <span class="lbl" style="left:8px;bottom:6px">↓ down</span>
        <div class="meas" id="pad-meas"></div><div class="target" id="pad-target"></div>
      </div>
      <div class="sl"><b>Yaw</b><input id="yaw" type="range" step="0.5"><span class="val mono"><span id="yawv"></span> <i id="yawm"></i></span></div>
      <div class="sl"><b>Pitch</b><input id="pitch" type="range" step="0.5"><span class="val mono"><span id="pitchv"></span> <i id="pitchm"></i></span></div>
      <div class="btns" id="presets"></div>
      <div class="note">Sim joint angles: yaw + = left, pitch + = down. Straight + level = yaw <span class="mono" id="refy"></span>°, pitch <span class="mono" id="refp"></span>°. Ring = target, dot = measured.</div>
      <div class="status" id="headst"></div>
    </div>

    <div class="card">
      <h2>Grippers <small>100% = fully open</small></h2>
      <div class="grip" id="g-left"></div>
      <div class="grip" id="g-right"></div>
      <div class="btns"><button class="primary" onclick="grip('left',100);grip('right',100)">Open both</button><button onclick="grip('left',0);grip('right',0)">Close both</button></div>
      <div class="status" id="gripst"></div>
    </div>

    <div class="card">
      <h2>Body <small id="body-age"></small></h2>
      <div class="stats">
        <div class="stat"><div class="k">Roll</div><div class="v mono" id="roll">–</div></div>
        <div class="stat"><div class="k">Pitch</div><div class="v mono" id="bpitch">–</div></div>
        <div class="stat"><div class="k">Hottest motor</div><div class="v mono" id="temp">–</div></div>
        <div class="stat"><div class="k">Camera sync</div><div class="v mono" id="spread">–</div></div>
      </div>
      <div class="horizon"><div class="line" id="hz"></div><div class="cross"></div></div>
      <div class="note">The body is driven by run_g1_control_loop.py (keyboard / policy).</div>
    </div>
  </section>
</div>
<script>
const CAL = __CALIB__, Y = CAL.joint_yaw, P = CAL.joint_pitch;
const $ = id => document.getElementById(id);
const yaw = $("yaw"), pitch = $("pitch"), pad = $("pad");
for (const [el, c] of [[yaw, Y], [pitch, P]]) { el.min = c.min_angle; el.max = c.max_angle; el.value = c.home_angle; }
$("refy").textContent = Y.ref_angle.toFixed(1); $("refp").textContent = P.ref_angle.toFixed(1);

// pad: x = yaw (left edge = max yaw, i.e. looking left), y = pitch (top = min = up)
const toPad = (y, p) => [100 * (Y.max_angle - y) / (Y.max_angle - Y.min_angle), 100 * (p - P.min_angle) / (P.max_angle - P.min_angle)];
const fromPad = (fx, fy) => [Y.max_angle - fx * (Y.max_angle - Y.min_angle), P.min_angle + fy * (P.max_angle - P.min_angle)];
const place = (el, y, p) => { const [x, z] = toPad(y, p); el.style.left = x + "%"; el.style.top = z + "%"; };
{ const [x, z] = toPad(Y.ref_angle, P.ref_angle); $("gl-v").style.left = x + "%"; $("gl-h").style.top = z + "%"; }

function post(path, body) { fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).catch(() => {}); }
function throttled(fn, ms) { let last = 0, timer = null; return () => { const now = Date.now();
  if (now - last >= ms) { last = now; fn(); } else if (!timer) timer = setTimeout(() => { timer = null; last = Date.now(); fn(); }, ms); }; }
const sendHead = throttled(() => post("/head", { yaw: +yaw.value, pitch: +pitch.value }), 50);
function showHead() { $("yawv").textContent = (+yaw.value).toFixed(1) + "°"; $("pitchv").textContent = (+pitch.value).toFixed(1) + "°"; place($("pad-target"), +yaw.value, +pitch.value); }
let headSynced = false;  // the target starts at the measured pose, not at home
function setHead(y, p) { headSynced = true; yaw.value = y; pitch.value = p; showHead(); sendHead(); }
for (const el of [yaw, pitch]) el.addEventListener("input", () => { headSynced = true; showHead(); sendHead(); });
let dragging = false;
function padEvent(e) { const r = pad.getBoundingClientRect();
  const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
  const [y, p] = fromPad(fx, fy); setHead(y, p); }
pad.addEventListener("pointerdown", e => { dragging = true; pad.setPointerCapture(e.pointerId); padEvent(e); });
pad.addEventListener("pointermove", e => { if (dragging) padEvent(e); });
pad.addEventListener("pointerup", () => { dragging = false; });
const presets = { "Home": [Y.home_angle, P.home_angle], "Straight + level": [Y.ref_angle, P.ref_angle],
  "Look left": [Y.ref_angle + 30, P.home_angle], "Look right": [Y.ref_angle - 30, P.home_angle], "Look down": [Y.ref_angle, P.home_angle + 20] };
for (const [name, [y, p]] of Object.entries(presets)) {
  const b = document.createElement("button"); b.textContent = name; if (name === "Home") b.className = "primary";
  b.onclick = () => setHead(y, p); $("presets").appendChild(b);
}
showHead();

const touched = { left: false, right: false };
for (const side of ["left", "right"]) {
  const name = side[0].toUpperCase() + side.slice(1);
  $("g-" + side).innerHTML = `<div class="grip-top"><b>${name}</b><span class="big mono" id="${side}-big">–</span></div>
    <div class="bar"><div id="${side}-bar"></div></div>
    <input id="${side}" type="range" min="0" max="100" step="1">
    <div class="btns"><button onclick="grip('${side}',100)">Open</button><button onclick="grip('${side}',50)">Half</button><button onclick="grip('${side}',0)">Close</button>
    <span class="val mono" style="margin-left:auto;align-self:center">cmd <span id="${side}-cmd">–</span></span></div>`;
  const el = $(side), send = throttled(() => post("/gripper", { side, open: +el.value }), 50);
  el.addEventListener("input", () => { touched[side] = true; $(side + "-cmd").textContent = el.value + "%"; send(); });
}
function grip(side, pct) { touched[side] = true; $(side).value = pct; $(side + "-cmd").textContent = pct + "%"; post("/gripper", { side, open: pct }); }

const fresh = a => a !== null && a !== undefined && a < 1.0;
const dot = (id, state) => { $(id).className = "dot " + state; };
async function poll() {
  try {
    const s = await (await fetch("/state")).json();
    const h = s.head, b = s.body, c = s.cameras;
    if (h && !headSynced && fresh(h.age)) { headSynced = true; yaw.value = h.yaw; pitch.value = h.pitch; showHead(); }
    if (h) { $("yawm").textContent = "(" + h.yaw.toFixed(1) + ")"; $("pitchm").textContent = "(" + h.pitch.toFixed(1) + ")"; place($("pad-meas"), h.yaw, h.pitch); }
    dot("p-head", fresh(h && h.age) ? "ok" : "bad");
    $("headst").textContent = fresh(h && h.age) ? "" : h ? "STALE: no head/state for " + h.age.toFixed(0) + " s (node down or USB lost)" : "no head/state: is head_control_node running?";
    $("headst").className = "status bad";
    let gripOk = true;
    for (const side of ["left", "right"]) {
      const g = s.grippers[side];
      if (!g || !fresh(g.age)) { gripOk = false; continue; }
      $(side + "-big").textContent = g.open.toFixed(0) + "%"; $(side + "-bar").style.width = Math.max(0, Math.min(100, g.open)) + "%";
      if (!touched[side]) $(side).value = g.open;
    }
    dot("p-grip", gripOk ? "ok" : "bad");
    $("gripst").textContent = gripOk ? "" : "no gripper state: is dex1_1_gripper_server running?"; $("gripst").className = "status bad";
    dot("p-body", fresh(b && b.age) ? "ok" : "bad");
    const looping = fresh(s.lowcmd_age); dot("p-loop", looping ? "ok" : "warn");
    $("p-loop-t").textContent = looping ? "Control loop on" : "Control loop off";
    if (b) { $("roll").textContent = b.roll.toFixed(1) + "°"; $("bpitch").textContent = b.pitch.toFixed(1) + "°"; $("temp").textContent = b.temp + "°C";
             $("hz").style.transform = `translateY(${Math.max(-40, Math.min(40, b.pitch * 2))}px) rotate(${-b.roll}deg)`;
             $("body-age").textContent = fresh(b.age) ? "live" : "stale"; }
    const camOk = fresh(c.age); dot("p-cams", camOk ? (c.fps > 25 ? "ok" : "warn") : "bad");
    $("p-cams-t").textContent = camOk ? c.fps.toFixed(1) + " fps" : "off";
    $("spread").textContent = camOk ? c.spread_ms.toFixed(1) + " ms" : "–";
    $("hud").textContent = camOk ? `${c.fps.toFixed(1)} fps · sync ${c.spread_ms.toFixed(1)} ms` : "no camera sets (head_zmq_bridge running?)";
  } catch (e) { for (const id of ["p-body", "p-head", "p-grip", "p-cams"]) dot(id, "bad"); }
  setTimeout(poll, 200);
}
poll();
</script></body></html>
"""


def make_handler(robot: Robot, calib: dict):
    page = PAGE.replace("__CALIB__", json.dumps(calib)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self, camera):
            """MJPEG: the latest frame of `camera`, at most STREAM_FPS, until the browser disconnects."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_id = -1
            try:
                while True:
                    with robot.new_frame:
                        robot.new_frame.wait_for(lambda: robot.frame_id != last_id, timeout=2.0)
                        last_id, jpg = robot.frame_id, robot.frames.get(camera)
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/state":
                self.reply(200, json.dumps(robot.snapshot()).encode(), "application/json")
            elif self.path.startswith("/stream/") and self.path[len("/stream/"):] in CAMERAS:
                self.stream(self.path[len("/stream/"):])
            else:
                self.reply(404, b"not found")

        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/head":
                    robot.send_head(float(data["yaw"]), float(data["pitch"]))
                elif self.path == "/gripper" and data.get("side") in ("left", "right"):
                    robot.send_gripper(data["side"], float(data["open"]))
                else:
                    return self.reply(404, b"not found")
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, str(e).encode())
            self.reply(204, b"")

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True  # camera streams end with the server


def main(args: Args):
    ChannelFactoryInitialize(0, args.interface)
    rclpy.init()
    robot = Robot(args.camera_url)
    threading.Thread(target=rclpy.spin, args=(robot,), daemon=True).start()
    server = Server(("0.0.0.0", args.port), make_handler(robot, load_head_calib(args.head_calib)))
    # rclpy's SIGTERM handler only shuts the ROS context down; also stop serving (plain `kill`)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"G1 control page on http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Exit without interpreter teardown: unitree_sdk2py's CycloneDDS threads are still running then, and their
        # C++ side aborts ("terminate called without an active exception"), leaving a core dump in the cwd.
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
