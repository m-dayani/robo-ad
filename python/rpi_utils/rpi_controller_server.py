"""
    Raspberry Pi Controller Server:
        - Send sensor data and images
        - Respond to control commands (control channel with client)
        - Update the PWM outputs

    Notes
        1. Always use image compression before send (e.g. JPEG) for enhanced frame rate
"""

import argparse
import sys
import threading

import numpy as np

from rpi_camera import RpiCamera
from rpi_pwm import RpiPwm

sys.path.append('../')
sys.path.append('../wireless/')
from wireless.wl_server import ControlServer
from wireless.image_server import ImageServer


class CmdReceiver(ControlServer):

    def __init__(self, host, port):
        super().__init__(host, port)

        self.v_vel = np.zeros(4, dtype=np.float32)
        self.v_pb = np.zeros(4, dtype=np.bool_)
        self.v_sw = np.zeros(4, dtype=np.bool_)

        self.pwm_generator = RpiPwm()
        self.show_time = False
        self.show_state = True

    def process_cmd(self, label, data):
        if len(label) < 0:
            print('label is empty: ' + label)
            return False

        if label == 'stop':
            print('Stop signal received from remote controller, breaking control loop')
            return False

        if label == 'state':
            try:
                # update control state
                state_vec = np.float32(data.split(','))
                if len(state_vec) >= 12:
                    self.v_vel = state_vec[:4]
                    self.v_pb = np.bool_(state_vec[4:8])
                    self.v_sw = np.bool_(state_vec[8:])
                    self.update_control()
                    if self.show_state:
                        print('remote controller: ' + str(state_vec))
            except ValueError:
                print('state conversion error: ' + str(data))
        return True

    def update_control(self):
        if len(self.v_vel) == 4:
            pitch, roll, trot, yaw = self.v_vel
            # trot = 120.0
            self.pwm_generator.update_analog(0, trot)
        else:
            print('Bad v_vel of size ' + str(len(self.v_vel)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Emulate a remote robot server (like RPI or ESP32-cam). 
        ''')
    parser.add_argument('-data_path', help='path to root video folder', default='.')
    parser.add_argument('--server_ip', help='server ip address', default='127.0.1.1')
    parser.add_argument('--server_port_ctrl', help='server port for control commands', default=29456)
    parser.add_argument('--server_port_data', help='server port for video stream', default=8080)
    parser.add_argument('--stream_mode',
                        help='stream mode: 0: async no-ack, 1: sync 1 ack, 2: sync 2 ack, 3: esp32', default=0)
    args = parser.parse_args()

    host = args.server_ip
    port = int(args.server_port_ctrl)
    my_server = CmdReceiver(host, port)

    image_loader = RpiCamera(img_cf='jpg')
    port_data = int(args.server_port_data)
    video_server = ImageServer(host, port_data, image_loader, args.stream_mode)
    image_loader.play_in_background()

    # run both servers in parallel
    thread_ctrl_server = threading.Thread(target=my_server.run)
    thread_data_server = threading.Thread(target=video_server.run)

    thread_ctrl_server.start()
    thread_data_server.start()

    thread_ctrl_server.join()
    video_server.should_stop = True
    thread_data_server.join(5)

    my_server.close()
    video_server.close()
    image_loader.stop()
    image_loader.close()


