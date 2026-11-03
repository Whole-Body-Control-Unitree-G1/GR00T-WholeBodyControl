"""Standalone script: subscribes to ZMQ camera frames and streams to PICO via TCP."""

import argparse
import fcntl
import socket
import struct
import termios
import threading
import time

import av
import cv2
import msgpack
import numpy as np
import zmq

ZMQ_PORT     = 5559
CAMERA_KEY   = 'stereo_view'
COMMAND_PORT = 13579
CAMERA_PORT  = 12345
BITRATE      = 4_000_000
FPS          = 60
# A frame is skipped (not encoded) while more than this many bytes still wait in the socket to the PICO: a slow wifi
# link then lowers the frame rate instead of building up seconds of lag in the send queue.
MAX_BACKLOG  = 64_000

# Set by command handler when PICO sends OPEN_CAMERA
pico_ip   = None
pico_lock = threading.Lock()


def parse_open_camera(data: bytes) -> str | None:
    """Extract PICO IP from OPEN_CAMERA packet by scanning for an IPv4 pattern."""
    import re
    match = re.search(rb'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', data)
    if match:
        return match.group(1).decode('ascii')
    print(f'[CMD] Failed to parse IP: no IPv4 pattern found in packet (raw: {data!r})')
    return None


def command_server():
    """Listen for OPEN_CAMERA command from PICO on port 13579."""
    global pico_ip
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', COMMAND_PORT))
    srv.listen(1)
    print(f'Command server listening on :{COMMAND_PORT}')
    while True:
        conn, addr = srv.accept()
        print(f'PICO connected from {addr}')
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                if b'OPEN_CAMERA' in data:
                    ip = parse_open_camera(data)
                    if ip:
                        print(f'OPEN_CAMERA received — PICO camera at {ip}:{CAMERA_PORT}')
                        with pico_lock:
                            pico_ip = ip
            except Exception:
                break
        print('PICO command disconnected')
        with pico_lock:
            pico_ip = None
        conn.close()


def init_encoder(w, h, fps=FPS, bitrate=BITRATE):
    container = av.open('/dev/null', 'w', format='null')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    stream.bit_rate = bitrate
    stream.options = {
        'preset': 'ultrafast',
        'tune': 'zerolatency',
        'profile': 'baseline',
    }
    return container, stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zmq-host', default='192.168.123.164', help='ZMQ camera bridge host (robot IP)')
    parser.add_argument('--fps', type=int, default=FPS,
                        help='the ZED stereo rate; the encoder spreads --bitrate over this many frames per second')
    parser.add_argument('--bitrate', type=int, default=BITRATE, help='H.264 bitrate (bit/s)')
    args = parser.parse_args()

    global pico_ip
    threading.Thread(target=command_server, daemon=True).start()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, True)
    sock.connect(f'tcp://{args.zmq_host}:{ZMQ_PORT}')
    print(f'Subscribed to tcp://{args.zmq_host}:{ZMQ_PORT}')
    print('Waiting for PICO OPEN_CAMERA command...')

    container = None
    stream    = None
    cam_sock  = None
    sent, skipped, last_report = 0, 0, time.monotonic()

    while True:
        raw = sock.recv()

        with pico_lock:
            ip = pico_ip

        if ip is None:
            continue

        # Connect to PICO camera port if not connected
        if cam_sock is None:
            try:
                cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cam_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send each frame at once
                cam_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * MAX_BACKLOG)  # no deep queue
                cam_sock.connect((ip, CAMERA_PORT))
                print(f'Connected to PICO camera at {ip}:{CAMERA_PORT}')
            except Exception as e:
                print(f'Failed to connect to PICO: {e}')
                cam_sock = None
                time.sleep(1)
                continue

        # the wifi to the PICO is behind: drop this frame rather than queue it (queued frames = lag)
        backlog = struct.unpack('i', fcntl.ioctl(cam_sock.fileno(), termios.TIOCOUTQ, b'\0' * 4))[0]
        if backlog > MAX_BACKLOG:
            skipped += 1
            continue

        data = msgpack.unpackb(raw, raw=False)
        jpg_bytes = data['images'][CAMERA_KEY]
        frame_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        h, w = frame_bgr.shape[:2]
        if stream is None:
            container, stream = init_encoder(w, h, args.fps, args.bitrate)
            print(f'Encoder initialized: {w}x{h} @ {args.fps}fps, {args.bitrate / 1e6:.1f} Mbit/s')

        # BGR -> I420 in one OpenCV call (~2 ms at 2560x720; the rgb24 -> yuv420p reformat took ~25 ms, so the loop
        # fell behind the camera)
        yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
        av_frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')

        sent += 1
        if time.monotonic() - last_report > 10.0:
            print(f'{sent / (time.monotonic() - last_report):.1f} fps sent, {skipped} frames skipped (wifi behind)')
            sent, skipped, last_report = 0, 0, time.monotonic()

        for packet in stream.encode(av_frame):
            data_bytes = bytes(packet)
            out = struct.pack('>I', len(data_bytes)) + data_bytes
            try:
                cam_sock.sendall(out)
            except Exception as e:
                print(f'Send error: {e} — reconnecting')
                cam_sock.close()
                cam_sock = None
                stream   = None
                break


if __name__ == '__main__':
    main()
