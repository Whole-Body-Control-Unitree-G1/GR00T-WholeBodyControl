#!/usr/bin/env bash
# One tmux session for the whole G1 embodiment, run from the laptop (robot on ethernet).
#
#   decoupled_wbc/scripts/launch_g1.sh start    # preflight checks, then start everything, attach to the session
#   decoupled_wbc/scripts/launch_g1.sh status   # what is running (tmux, Orin processes, control page state)
#   decoupled_wbc/scripts/launch_g1.sh stop     # Ctrl+C every window, wait, close the session
#
# Windows (tmux session "g1"; switch with Ctrl+b <number>):
#   1 grippers  Orin: dex1_1_gripper_server (sudo: type the unitree password here once, unless sudoers allows it)
#   2 cameras   Orin: ~/robot_cameras/cameras.launch.py (ZED Mini + 2x D405 drivers, ROS on loopback, no web viewer)
#   3 bridge    Orin: head_zmq_bridge (time-aligned ego_view / left_wrist / right_wrist JPEG sets -> ZMQ :5555)
#   4 head      Orin: head_control_node (after the gripper server is up; moves the head to the sim home pose)
#   5 page      laptop container: robot_web.py control page -> http://localhost:8080
#   6 control   laptop container: run_g1_control_loop.py typed but NOT started: robot on the gantry, debug mode, then Enter
#   7 teleop    laptop container: PICO teleop loop (joysticks -> base, arm tracking -> upper body, headset -> head),
#               typed but NOT started: after the control loop is up and the headset streams (XRoboToolkit app)
#   8 record    laptop container: record_teleop_homie.py -> $DATA/$TASK (PICO A start / save, B discard), typed
#   9 video     laptop container: gear_sonic zed_pico_zmq.py, the ZED stereo view (Orin bridge :5559) -> PICO headset
#               (H.264 on :12345 once the XRoboToolkit app asks for the camera); runs from the start
#  11 policy    laptop container: run_policy_homie.py (pi0.5 policy server on POLICY_HOST drives the robot instead of the
#               PICO teleop; window 7 must NOT run), typed but NOT started
#  10 voice     laptop: speaks the PICO-relevant events of windows 6-9 (teleop on / paused, recording, saved, ...),
#               since the operator wears the headset; their output is piped to /tmp/g1_<session>/
#
# Settings (environment): ROBOT, SESSION, CONTAINER, IMAGE, IFACE, ORIN_IFACE, HEADCTRL, DATA, TASK, PROMPT, HEIGHT,
# VOICE (0 = no voice window), VIDEO_BITRATE, POLICY_HOST, POLICY_DIR, OPENPI_CLIENT.
set -euo pipefail

ROBOT=${ROBOT:-unitree@192.168.123.164}
SESSION=${SESSION:-g1}
CONTAINER=${CONTAINER:-g1_wbc}
IMAGE=${IMAGE:-nvgear/gr00t_wbc:latest}
IFACE=${IFACE:-enp0s31f6}  # laptop port on the robot's 192.168.123.x network
ORIN_IFACE=${ORIN_IFACE:-enP8p1s0}  # Orin port on the same network
HEADCTRL=${HEADCTRL:-$HOME/wbcG1/headControl}
DATA=${DATA:-$HOME/wbcG1/real_data}  # raw teleop episodes (mounted into the container)
TASK=${TASK:-cube_right}
PROMPT=${PROMPT:-Pick up the rubiks cube.}
HEIGHT=${HEIGHT:-0.74}  # PICO teleop base height command at startup (the sim cube data mostly used 0.70)
VOICE=${VOICE:-1}
POLICY_HOST=${POLICY_HOST:-lamb.mech.northwestern.edu}  # openpi policy server (serve_policy_rtc.py, port 8000)
POLICY_DIR=${POLICY_DIR:-$HOME/wbcG1/policies/pi05_g1_cube_abs_30000}  # holds the checkpoint's norm_stats.json
OPENPI_CLIENT=${OPENPI_CLIENT:-$HOME/wbcG1/openpi/packages/openpi-client/src}  # openpi_client, mounted read-only
VIDEO_BITRATE=${VIDEO_BITRATE:-3000000}  # headset video (bit/s, at the ZED's 30 fps); lower it on a weak wifi
LOGDIR=/tmp/g1_${SESSION:-g1}  # windows 6-9 output, read by the voice window
REPO=$(cd "$(dirname "$0")/../.." && pwd)
U2D2=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNC7-if00-port0
SSH=(ssh -o ConnectTimeout=5 -o ServerAliveInterval=5)

# Orin process checks use [x]yz patterns so the ssh command line itself never matches.
ORIN_PROCS='[d]ex1_1_gripper_server|[h]ead_control_node|[r]ealsense2_camera_node|[z]ed_wrapper|[h]ead_zmq_bridge|[s]tart_cameras.sh|[c]ameras.launch.py|[c]am_web_viewer'

# ROS 2 between the Orin and the laptop runs over the robot ethernet only (CycloneDDS pinned to one interface per side;
# the Orin's ~/.bashrc default pins it to wifi). The camera drivers keep their own loopback-only config.
ORIN_CDDS='$HOME/wbcG1/cyclone_eth.xml'  # expanded on the Orin
LAPTOP_CDDS=/tmp/cyclone_eth.xml  # inside the container
ROS_ORIN='source /opt/ros/humble/setup.bash && source ~/wbcG1/headControl/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
ROS_LAPTOP='source /opt/ros/humble/setup.bash && source /root/venv/bin/activate && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///tmp/cyclone_eth.xml PYTHONPATH=$PWD:$PYTHONPATH'

say() { printf '\033[1;36m[g1]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[g1]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[g1]\033[0m %s\n' "$*" >&2; exit 1; }

orin() { "${SSH[@]}" -o BatchMode=yes "$ROBOT" "$@"; }

cdds_xml() {  # CycloneDDS config using only network interface $1
    printf '<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="%s"/></Interfaces></General></Domain></CycloneDDS>\n' "$1"
}

preflight() {
    command -v tmux >/dev/null || die "tmux is not installed (sudo apt install tmux)"
    tmux has-session -t "$SESSION" 2>/dev/null && die "tmux session '$SESSION' already exists: '$0 stop' first, or 'tmux attach -t $SESSION'"
    ip -4 -o addr show "$IFACE" 2>/dev/null | grep -q '192\.168\.123\.' \
        || die "$IFACE has no 192.168.123.x address: plug in the robot ethernet and set a static IP (e.g. 192.168.123.222/24)"
    orin true 2>/dev/null || die "cannot ssh to $ROBOT (robot on? key-based login set up?)"
    local running
    running=$(orin "ps -eo pid,args | grep -E '$ORIN_PROCS' | cut -c1-100" || true)
    [ -z "$running" ] || die "already running on the Orin (stop those first, or '$0 stop' if this launcher started them):
$running"
    orin "test -e $U2D2" || die "head U2D2 not found on the Orin ($U2D2): check its USB"
    orin "test -w $U2D2" || die "the Orin user cannot write the head port ($U2D2). Permanent fix on the Orin:
  sudo mv /etc/udev/rules.d/99-u2d2.rule /etc/udev/rules.d/99-u2d2.rules   # udev only reads *.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty
or for now: sudo chmod 666 \$(readlink -f $U2D2)"
    local serial
    serial=$(orin "ls /dev/serial/by-id/ | grep -c FTBRJ9EO" || true)
    [ "${serial:-0}" -ge 4 ] || warn "gripper serial hub (FT4232H FTBRJ9EO) not found on the Orin: grippers will not start"
    curl -s -m 1 -o /dev/null http://localhost:8080/ && die "port 8080 on the laptop is in use (an old control page?)"
    return 0
}

ensure_container() {
    mkdir -p "$DATA"
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
        say "starting container $CONTAINER ($IMAGE)"
        # --ulimit core=0: a crashing process would otherwise write a ~1 GB root-owned "core" into the repo (its cwd)
        mkdir -p "$POLICY_DIR"
        docker run -d --rm --name "$CONTAINER" --network host --ulimit core=0 -e PYTHONPATH="$REPO:/opt/openpi_client" \
            -v "$REPO:$REPO" -v "$HEADCTRL:$HEADCTRL:ro" -v "$DATA:$DATA" -v "$OPENPI_CLIENT:/opt/openpi_client:ro" \
            -v "$POLICY_DIR:$POLICY_DIR:ro" -w "$REPO" "$IMAGE" sleep infinity >/dev/null
    else
        local mounts
        mounts=$(docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$CONTAINER")
        echo "$mounts" | grep -qw "$DATA" \
            || die "container $CONTAINER was started without $DATA mounted (the recorder could not save): docker stop $CONTAINER"
        echo "$mounts" | grep -qw /opt/openpi_client \
            || die "container $CONTAINER was started without the openpi client (policy window): docker stop $CONTAINER"
    fi
    # the policy client's websocket dependency (not in the image)
    docker exec "$CONTAINER" bash -c 'source /root/venv/bin/activate && python -c "import websockets" 2>/dev/null || pip install -q websockets' \
        || warn "could not install websockets in the container: the policy window will not run"
}

# window <index> <name> <command> [run]: a tmux window running <command>; without "run" it is only typed.
window() {
    local idx=$1 name=$2 cmd=$3 run=${4:-run}
    if [ "$idx" = 1 ]; then
        tmux new-session -d -s "$SESSION" -n "$name" -x 220 -y 50
        tmux move-window -s "$SESSION:0" -t "$SESSION:1"  # number the windows 1-8
    else
        tmux new-window -t "$SESSION:$idx" -n "$name"
    fi
    tmux send-keys -t "$SESSION:$idx" "$cmd"
    [ "$run" = run ] && tmux send-keys -t "$SESSION:$idx" Enter
    return 0
}

start() {
    preflight
    ensure_container
    cdds_xml "$ORIN_IFACE" | orin "cat > $ORIN_CDDS"
    cdds_xml "$IFACE" | docker exec -i "$CONTAINER" sh -c "cat > $LAPTOP_CDDS"
    local ssh_t="${SSH[*]} -t $ROBOT"
    local calib="$HEADCTRL/src/headctrl/head_control/config/head_joint_calib.yaml"

    say "starting tmux session '$SESSION'"
    window 1 grippers "$ssh_t 'cd ~/dex1_1_service/bin && sudo ./dex1_1_gripper_server --network enP8p1s0'"
    # the camera drivers only (start_cameras.sh would also start its web viewer on the wifi address)
    window 2 cameras "$ssh_t 'source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://\$HOME/robot_cameras/cyclone_localhost.xml && exec ros2 launch ~/robot_cameras/cameras.launch.py'"
    # the bridge subscribes on loopback like the camera drivers; give them time to come up
    window 3 bridge "$ssh_t '$ROS_ORIN && export CYCLONEDDS_URI=file://\$HOME/robot_cameras/cyclone_localhost.xml && sleep 10 && exec ros2 run head_zmq_bridge head_zmq_bridge --ros-args --params-file ~/wbcG1/headControl/install/head_zmq_bridge/share/head_zmq_bridge/config/camera.yaml'"
    # head: ROS over the robot ethernet (ORIN_CDDS).
    # It starts only after the gripper server has found its motors: that probe opens every /dev/ttyUSB*, the head's
    # U2D2 included, and re-configures the port under a head node that is already running (it then never reads the
    # motors again). Up to 2 min, so there is time to type the sudo password in window 1.
    local wait_grippers="echo 'waiting for the gripper server (window 1) to finish probing the serial ports...'; for i in \$(seq 120); do tmux capture-pane -p -S -200 -t $SESSION:1 | grep -q 'Gripper Server started' && break; sleep 1; done"
    window 4 head "$wait_grippers; $ssh_t '$ROS_ORIN && export CYCLONEDDS_URI=file://$ORIN_CDDS && exec ros2 launch head_control head_control.launch.xml'"
    window 5 page "docker exec -it $CONTAINER bash -c '$ROS_LAPTOP && exec python decoupled_wbc/scripts/robot_web.py --interface $IFACE --head-calib $calib'"
    window 6 control "docker exec -it $CONTAINER bash -c '$ROS_LAPTOP && exec python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real --waist-location upper_body --stand-cmd-threshold 0.1 --no-with-hands --with-dex1-grippers --no-enable-onscreen --enable-gravity-compensation --gravity-compensation-joints arms waist'" typed
    # PICO: the XRoboToolkit PC service (started by the streamer) talks to the headset app; --pico-head-tracking makes
    # the robot head follow the headset (zero pose: first frame, re-zero with a right stick click); --pico-pause-home:
    # pausing teleop (left menu + right trigger) homes the arms, opens the grippers and levels the head, joysticks still walk
    window 7 teleop "docker exec -it $CONTAINER bash -c '$ROS_LAPTOP && exec python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py --body-control-device pico --hand-control-device pico --waist-location upper_body --pico-head-tracking --pico-pause-home --pico-base-height $HEIGHT'" typed
    # the container runs as root: hand the episodes back to the laptop user when the recorder exits
    window 8 record "docker exec -it $CONTAINER bash -c '$ROS_LAPTOP && python decoupled_wbc/scripts/record_teleop_homie.py --out $DATA/$TASK --prompt \"$PROMPT\" --interface $IFACE; chown -R $(id -u):$(id -g) $DATA'" typed
    # the headset view: waits for the PICO app's OPEN_CAMERA (command port 13579) and the Orin's stereo_view (:5559)
    window 9 video "docker exec -it $CONTAINER bash -c 'source /root/venv/bin/activate && exec python gear_sonic/scripts/zed_pico_zmq.py --zmq-host ${ROBOT#*@} --fps 30 --bitrate $VIDEO_BITRATE'"
    rm -rf "$LOGDIR" && mkdir -p "$LOGDIR"
    for w in 6 7 8 9; do
        touch "$LOGDIR/$w.log"
        tmux pipe-pane -t "$SESSION:$w" -o "cat >> $LOGDIR/$w.log"
    done
    # policy instead of teleop: stop window 7 first (both publish the control goal)
    window 11 policy "docker exec -it $CONTAINER bash -c '$ROS_LAPTOP && exec python decoupled_wbc/scripts/run_policy_homie.py --policy-host $POLICY_HOST --norm-stats $POLICY_DIR/norm_stats.json --prompt \"$PROMPT\" --interface $IFACE --out $DATA/policy_\$(date +%m%d_%H%M%S).npz'" typed
    [ "$VOICE" = 1 ] && window 10 voice "SESSION=$SESSION $(realpath "$0") voice"
    tmux select-window -t "$SESSION:1"

    say "started. Next:"
    echo "   - window 1 (grippers): type the unitree sudo password if asked"
    echo "   - control page: http://localhost:8080 (all status lights green after ~20 s)"
    echo "   - window 6 (control): robot hanging on the gantry in debug mode, then press Enter there"
    echo "   - window 7 (teleop): PICO app streaming, look straight ahead, Enter; right stick click = re-zero the head"
    echo "     left menu + right trigger = teleop on / pause (pause: arms home, grippers open, head level, joysticks walk)"
    echo "   - window 8 (record): Enter; PICO A = start / save an episode, B = discard -> $DATA/$TASK"
    echo "     save (A) BEFORE pausing, else the homing move is recorded"
    echo "   - window 9 (video): starts by itself; open the camera in the PICO app -> 'Connected to PICO camera'"
    echo "   - window 10 (voice): speaks teleop on / paused / recording / saved / discarded / warnings"
    echo "   - window 11 (policy): INSTEAD of window 7; needs the policy server up on $POLICY_HOST:8000. Enter, then Enter"
    echo "     again to start; Ctrl+C holds the pose. Add --dry-run to the typed command to only print the policy's output"
    echo "   - switch windows: Ctrl+b <1-8>; detach: Ctrl+b d; stop everything: $0 stop"
    if [ -t 1 ]; then tmux attach -t "$SESSION"; fi
}

status() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        say "tmux session '$SESSION':"
        tmux list-windows -t "$SESSION" -F '   #{window_index} #{window_name}  (#{pane_current_command})'
    else
        say "no tmux session '$SESSION'"
    fi
    say "Orin processes:"
    orin "ps -eo pid,etime,args | grep -E '$ORIN_PROCS' | cut -c1-100" 2>/dev/null | sed 's/^/   /' || warn "Orin unreachable"
    say "control page:"
    curl -s -m 2 http://localhost:8080/state | python3 -c '
import json, sys
s = json.load(sys.stdin)
fresh = lambda a: a is not None and a < 1.0
print("   head     ", "ok" if s["head"] and fresh(s["head"]["age"]) else "NO STATE")
print("   grippers ", {k: "ok" if fresh(g["age"]) else "stale" for k, g in s["grippers"].items()} or "NO STATE")
print("   body     ", "ok" if s["body"] and fresh(s["body"]["age"]) else "NO STATE", "| control loop", "on" if fresh(s["lowcmd_age"]) else "off")
c = s["cameras"]; print("   cameras  ", f"{c['fps']:.1f} fps" if fresh(c["age"]) else "NO FRAMES")' 2>/dev/null || echo "   not reachable"
    if timeout 1 bash -c "</dev/tcp/${ROBOT#*@}/5559" 2>/dev/null; then
        say "PICO video source: Orin stereo_view on :5559 is up"
    else
        warn "PICO video source: nothing on ${ROBOT#*@}:5559 (the bridge does not publish stereo_view; headset gets no image)"
    fi
}

voice() {  # window 10: speak the PICO-relevant events of windows 6-9 (piped to $LOGDIR by start)
    local last="" last_t=0 msg n
    if command -v spd-say >/dev/null; then
        speak() { spd-say -w -- "$1"; }
    elif command -v espeak >/dev/null; then
        speak() { espeak -- "$1" 2>/dev/null; }
    else
        die "no spd-say or espeak: sudo apt install speech-dispatcher"
    fi
    say "speaking the events of windows 6-9 ($LOGDIR)"
    speak "voice on"
    tail -n 0 -q -F "$LOGDIR"/6.log "$LOGDIR"/7.log "$LOGDIR"/8.log "$LOGDIR"/9.log 2>/dev/null \
        | stdbuf -oL tr '\r' '\n' | sed -u 's/\x1b\[[0-9;?]*[A-Za-z]//g' | while IFS= read -r line; do
        msg=""
        case "$line" in
            *"[active] teleop on"*) msg="teleop on" ;;
            *"[pause] homing"*) msg="paused" ;;
            *"head zero pose calibrated"*) msg="head zeroed" ;;
            *">>> episode "*": recording"*) n=${line#*>>> episode }; msg="recording episode ${n%%:*}" ;;
            *"saved episode_"*) n=${line#*saved episode_}; msg="saved episode $((10#${n%%:*}))" ;;
            *"shorter than 1 s: discarded"*) msg="too short, discarded" ;;
            *"longer than"*"discarded"*) msg="too long, discarded" ;;
            *"episode discarded"*) msg="discarded" ;;
            *"cannot start, no data yet"*) msg="cannot record, no camera data" ;;
            *"SAFETY: Teleop mode timeout"*) msg="warning, teleop lost" ;;
            *"Connected to PICO camera"*) msg="video connected" ;;
            *"Send error"*) msg="video lost" ;;
        esac
        # the same message within 3 s is said once (e.g. a repeated warning)
        if [ -n "$msg" ] && { [ "$msg" != "$last" ] || [ $((SECONDS - last_t)) -ge 3 ]; }; then
            echo "$(date +%T)  $msg"
            speak "$msg"
            last=$msg last_t=$SECONDS
        fi
    done
}

stop() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        say "Ctrl+C in every window (recorder and teleop first, then the control loop)"
        for w in 11 10 9 8 7 6 5 4 3 2 1; do tmux send-keys -t "$SESSION:$w" C-c 2>/dev/null || true; done
        for _ in $(seq 1 15); do
            [ -z "$(orin "ps -eo args | grep -E '$ORIN_PROCS'" 2>/dev/null)" ] && break
            sleep 1
        done
        tmux kill-session -t "$SESSION"
    else
        say "no tmux session '$SESSION'"
    fi
    local left
    left=$(orin "ps -eo pid,args | grep -E '$ORIN_PROCS' | cut -c1-100" 2>/dev/null || true)
    if [ -n "$left" ]; then
        warn "still running on the Orin (stop by hand, e.g. ~/robot_cameras/stop_cameras.sh, or kill <pid>):"
        echo "$left"
    else
        say "Orin: all stopped"
    fi
    say "the container $CONTAINER keeps running (docker stop $CONTAINER to remove it)"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    voice) voice ;;
    *) echo "usage: $0 start|status|stop"; exit 1 ;;
esac
