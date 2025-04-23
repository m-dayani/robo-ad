"""
    Controller client:
        Operate input (joystick, keyboard, ...)
        Send control state to remote robot (wireless)
        Get data from remote server (sensor and images)

    Notes:
        Handling image stream from RPI is different from a ESP32-Cam!
        In either case, although it can show the image in the main control loop,
        this operation slows down input sensitivity drastically!
        Better to see frame in a browser instead (in case of ESP32)
        Both pygame main control loop and imshow must run on the main thread

        received images are usually low-quality with large latency and fps
            (around 3fps from RPI) even in no_ack mode
"""

import sys
import threading
import time
import argparse

import numpy as np
import cv2

sys.path.append('../')
sys.path.append('../wireless/')
from man_ctrl_pygame import MyJoystick
from wireless.wl_socks import MyWirelessDriver
from wireless.image_client import ImageClient
from wireless.wl_client import ControlClient


class CmdSender(ControlClient):
    def __init__(self, host, port, send_delay=0.01):
        super().__init__(host, port, send_delay)
        self.state = np.zeros(12, dtype=np.float32)

    def update_state(self, v_vel, v_pb, v_sw):
        self.state = np.hstack((v_vel, v_pb, v_sw))

    def serialize_state(self):
        payload = MyWirelessDriver.array_to_str(self.state, 2)
        return MyWirelessDriver.encode('state', payload)


class ImageReceiver(ImageClient):
    def __init__(self, host, port, stream_mode=0, img_cf='numpy'):
        super().__init__(host, port, stream_mode)
        self.img_cf = img_cf

    def process_img_bytes(self, img_bytes, img_size):
        recv_image = np.frombuffer(img_bytes, np.uint8)
        if self.img_cf == 'png' or self.img_cf == 'jpg':
            self.frame = cv2.imdecode(recv_image, cv2.IMREAD_UNCHANGED)
        else:
            self.frame = recv_image.reshape(img_size)

        cv2.imshow('Received Frames', self.frame)
        cv2.waitKey(1)

    def stop_stream(self):
        self.should_stop = True
        self.esp32_stream.stop_stream = True


class FlightController(MyJoystick):
    def __init__(self, host, port, port_vs=8080, stream_mode=0, img_cf='numpy', send_delay=0.01, show_frames=False):
        super().__init__()
        self.cmd_client = CmdSender(host, port, send_delay)
        self.cmd_client.connect()
        self.data_client = ImageReceiver(host, port_vs, stream_mode, img_cf)
        self.data_client.connect()

        self.frame = None
        self.show_frames = show_frames

        self.last_t_ctrl = -1
        self.last_t_stream = -1
        self.show_time = False

    def update(self):
        super().update()
        self.cmd_client.update_state(self.v_vel, self.v_pb, self.v_sw)
        if self.stop_man_ctrl:
            self.cmd_client.should_stop = True
            self.data_client.stop_stream()

    def show_loop_time(self, msg, last_time):
        t_curr = time.time_ns()
        if self.show_time and last_time > 0:
            t_diff = (t_curr - last_time) * 1e-6
            print(msg + str(t_diff) + ' ms')
        return t_curr

    def show_loop_time2(self, t_start, n_data):
        if self.show_time:
            t_end = time.time_ns()
            print('processed: ' + str(n_data) + ' bytes in ' + str((t_end - t_start) * 1e-6) + ' ms')


if __name__ == "__main__":

    # You can retrieve the necessary arguments from cl, config file, or hard-coded here
    host = '127.0.1.1'
    port_ctrl = 29456
    port_data = 8080

    parser = argparse.ArgumentParser(description='''
            Controller Client to control remote robots and devices (like RPI or ESP32-cam). 
            ''')
    parser.add_argument('--host_ip', help='server ip address', default=host)
    parser.add_argument('--port_ctrl', help='server port for control commands', default=port_ctrl)
    parser.add_argument('--port_data', help='server port for video stream', default=port_data)
    parser.add_argument('--stream_mode',
                        help='stream mode: 0: async no-ack, 1: sync 1 ack, 2: sync 2 ack, 3: esp32', default=0)
    parser.add_argument('--show_frames', help='show_frames (default: 1)', default=1)
    parser.add_argument('--img_cf', help='received image encoding format: numpy, jpg (default), png', default='jpg')
    parser.add_argument('--use_image_server', help='use image server (default: True)', default=1)
    args = parser.parse_args()

    host = args.host_ip
    port_ctrl = int(args.port_ctrl)
    port_data = int(args.port_data)
    stream_mode = int(args.stream_mode)
    show_frames = int(args.show_frames) == 1
    use_image_server = int(args.use_image_server) == 1

    send_delay = 0.1    # 0.01
    if stream_mode == 3:
        send_delay = 0.1
    flight_controller = FlightController(host, port_ctrl, port_data, stream_mode, send_delay=send_delay,
                                         img_cf=args.img_cf, show_frames=show_frames)

    thread_control = threading.Thread(target=flight_controller.cmd_client.send_control)
    # flight_controller.cmd_client.connect()
    thread_control.start()

    thread_video = None
    if use_image_server:
        thread_video = threading.Thread(target=flight_controller.data_client.run)
        # flight_controller.data_client.connect()
        # time.sleep(2)
        thread_video.start()

    # pygame control must be in the main thread
    flight_controller.run()

    flight_controller.stop_man_ctrl = True
    thread_control.join(5)
    if thread_video is not None:
        thread_video.join(5)

    flight_controller.clean()

