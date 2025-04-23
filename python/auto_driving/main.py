"""
    Autonomous Driving System

    Sensors:
        Ultrasonic (GPIO)
        Camera

    Communications:
        Serial (Arduino/ESP32)
        WiFi (Data and Command Transmission)

    Operation:
        Manual: receives commands from remote controller
        Auto:   detects objects, reads sensors, image processing to determine the right command
"""

import argparse
import copy
import sys
import threading
import time
import signal
from subprocess import check_output

import numpy as np

sys.path.append('../')
sys.path.append('../wireless/')
sys.path.append('../rpi_utils/')
sys.path.append('../data_loader/')
from wireless.wl_server import ControlServer
from wireless.image_server import ImageServer
from my_serial.arduino import MyArduino
from auto_driving import AutoDriving
import tools.utils as fst
from rpi_utils.rpi_camera import RpiCamera
from rpi_utils.rpi_ultrasonic import UltraSonic
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


instructions = '''
==================== Auto-Driving Instructions ====================
Q_KEY:              Tack Pictures
Num_KEY 1:          ON: Auto-Driving (Race), OFF: Manual
Num_KEY 2:          ON: Auto-Driving (Urban), OFF: Manual
UP/DOWN Arrow:      Trottle in manual control
LEFT/RIGHT Arrow:   Steering in manual control
ESC:                Stop and quit
===================================================================
'''


class CmdReceiver(ControlServer):

    def __init__(self, host, port):
        super().__init__(host, port)
        self.state_vec = np.zeros(12, dtype=np.float32)
        self.state_vec[:2] = 127.5
        self.state_vec_lock = threading.Lock()

    def process_cmd(self, label, data):
        if len(label) < 0:
            my_logger.warning('CmdReceiver.process_cmd, label is empty: ' + label)
            return False

        if label == 'stop':
            print('Stop signal received from remote controller, breaking control loop')
            return False

        if label == 'state':
            try:
                # update control state
                state_vec = np.float32(data.split(','))
                self.update_state(state_vec)
            except ValueError:
                my_logger.debug('CmdReceiver.process_cmd, state conversion error: ' + str(data))
                pass
        return True

    def update_state(self, state_vec):
        self.state_vec_lock.acquire()
        self.state_vec = state_vec
        self.state_vec_lock.release()

    def get_last_state(self):
        self.state_vec_lock.acquire()
        state_vec = np.copy(self.state_vec)
        self.state_vec_lock.release()
        return state_vec


class MySystem(object):
    def __init__(self, settings, image_loader):

        self.image_loader = image_loader

        self.auto_driving = AutoDriving(settings)
        self.auto_started = False

        # Arduino: serial transmission between RPI server and microcontroller
        self.arduino = MyArduino(settings['serial_port'], settings['serial_bd'],
                                 get_state=self.serialize_out_state,
                                 send_delay=settings['serial_send_delay'])

        # CMD Server
        host = settings['wl_host']
        port = int(settings['wl_port'])
        self.cmd_server = CmdReceiver(host, port)

        # Video Server
        port_data = int(settings['wl_port_data'])
        self.video_server = ImageServer(host, port_data, settings['stream_mode'])

        self.client_connected = False
        self.thread_ctrl_server = None
        self.thread_data_server = None

        self.dist_sensor = UltraSonic(use_sensor=(settings['use_ultrasonic'] == 1),
                                      loop_delay=settings['ultrasonic_delay'])
        self.dist_stop = False
        self.dist_cnt = 0
        self.dist_cnt_max = 15

        self.v_vel_default = np.zeros(4, dtype=np.float32)
        self.v_vel_default[:2] = 127.5
        self.v_vel = np.copy(self.v_vel_default)
        self.v_pb = np.zeros(4, dtype=np.bool_)
        self.v_sw = np.zeros(4, dtype=np.bool_)
        # out state: [f/b (trot), l/r (steering), obj, ...]
        self.out_state = np.zeros(8, dtype=np.uint8)
        self.out_state[:2] = np.uint8(self.v_vel_default[:2])

        self.lim_speed = [53, 203]
        self.lim_angle = [60, 120]

        self.sig_int = False

        # logging
        my_logger.info('auto_driving.main: server_ip is ' + str(host))
        # my_logger.info('auto_driving.main: created a dataset at ' + str(record_root))

    def run(self):
        signal.signal(signal.SIGINT, self.signal_int_handler)

        # listen for connections
        self.thread_ctrl_server = threading.Thread(target=self.cmd_server.run)
        self.thread_data_server = threading.Thread(target=self.video_server.run)
        self.thread_ctrl_server.start()
        self.thread_data_server.start()

        print('Waiting for client to connect ...')
        while self.cmd_server.client_socket is None:
            if self.sig_int:
                return
            time.sleep(1)

        # self.image_loader.play_in_background()
        self.arduino.start_in_background()
        self.dist_sensor.run_in_background()

        # this flag is for reconnecting other clients, but it's not available now
        self.client_connected = True

        while True:
            # if any client is connected, start the main process
            # check cmd commands (stop, run, auto_driving, ...)
            self.check_control()

            if not self.client_connected:
                print('AD System: Client disconnected, breaking the main loop')
                break

            # get last image from image loader
            frame = self.image_loader.get_next()
            if frame is not None:
                # supply input image to the other components
                # (image server, image recorder, and auto-driving/ml-detector)
                compressed_frame = copy.deepcopy(frame)
                compressed_frame.frame = self.image_loader.compress(compressed_frame.frame)

                self.video_server.set_last_frame(compressed_frame)
                # for the last two, nothing happens if these are not started,
                # but in case of video_recorder, setting frame is necessary for single image capture
                # self.video_recorder.set_last_frame(frame)
                self.auto_driving.set_last_frame(frame)

            # decide what command to send to the arduino or record (from 3 sources)
            self.update_control()

        if self.auto_started:
            self.auto_driving.stop()
            self.auto_started = False

        self.dist_sensor.stop(True)
        self.arduino.stop()

    def clean(self):

        if self.auto_started:
            self.auto_driving.stop()
            self.auto_started = False

        self.dist_sensor.clean()
        self.arduino.close()
        self.auto_driving.clean()

        if self.thread_ctrl_server is not None:
            self.cmd_server.stop()
            self.thread_ctrl_server.join(5)
            self.thread_ctrl_server = None
            self.cmd_server.close()

        if self.thread_data_server is not None:
            self.video_server.stop()
            self.thread_data_server.join(5)
            self.thread_data_server = None
            self.video_server.close()

    def check_control(self):

        if not self.cmd_server.is_started():
            self.client_connected = False
            return

        # update the internal state
        state_vec = self.cmd_server.get_last_state()
        if len(state_vec) >= 12:
            self.v_vel = state_vec[:4]
            self.v_pb = np.bool_(state_vec[4:8])
            self.v_sw = np.bool_(state_vec[8:])
        else:
            my_logger.warning('AD_System.check_control: incorrect state vector: ' + str(state_vec))
            return

        # determine the driving mode
        if self.v_sw[0] or self.v_sw[1]:
            if not self.auto_started:
                if self.v_sw[1]:
                    self.auto_driving.set_urban(True)
                self.auto_driving.start_auto_driving()
                self.auto_started = True
                print('Auto Racing')
        else:
            if self.auto_started:
                self.auto_driving.stop()
                self.auto_started = False
                self.auto_driving.set_urban(False)
                print('Manual Control')

    def update_control(self):

        # self.out_state = np.uint8(self.v_vel_default)

        if self.auto_started:
            # update the velocity state
            self.out_state = self.auto_driving.get_state()
            # print(self.out_state)
        else:
            # use the joystick app to control the robot
            self.out_state = self.get_man_state()

        if self.auto_started:
            # Obstacle avoidance
            # check stop
            dist = self.dist_sensor.get_distance()
            if 0 <= dist < 10:
                # print("DANGER! Obstacle at distance = %.1f cm" % dist)
                self.dist_stop = True
                self.dist_cnt = 0
            else:
                self.dist_cnt += 1
                if self.dist_cnt > self.dist_cnt_max:
                    self.dist_stop = False
                    self.dist_cnt = 0
            # decision-making
            if self.dist_stop:
                self.out_state[0] = int(self.v_vel_default[0])

    def get_man_state(self):
        # convert the internal state to the serial state
        state = np.zeros_like(self.out_state)
        state[:2] = np.uint8(self.v_vel[:2])
        return state

    def check_out_state(self):
        arduino_state = np.copy(self.out_state)

        # trottle
        if len(arduino_state) >= 0:
            speed = arduino_state[0]
            if speed < self.lim_speed[0]:
                arduino_state[0] = self.lim_speed[0]
            if speed > self.lim_speed[1]:
                arduino_state[0] = self.lim_speed[1]

        # steering
        if len(arduino_state) >= 1:
            angle = arduino_state[1]
            # convert angles from 0~255 to 60~120
            angle = angle / 255 * abs(self.lim_angle[1] - self.lim_angle[0]) + self.lim_angle[0]
            # print('Angle: ' + str(angle))
            if angle < self.lim_angle[0]:
                angle = self.lim_angle[0]
            if angle > self.lim_angle[1]:
                angle = self.lim_angle[1]
            arduino_state[1] = angle

        return arduino_state

    def get_line(self):
        return str(int(time.time_ns())) + ',' + ','.join(map(str, self.out_state[:2])) + '\r\n'

    def serialize_out_state(self):
        arduino_state = self.check_out_state()
        msg = 'dir:' + ':'.join(map(str, np.uint8(arduino_state))) + ':'
        return msg

    def signal_int_handler(self, sig, frame):
        print('\nInterrupt Execution Thread!')
        # sys.exit(0)
        self.sig_int = True


if __name__ == "__main__":

    settings_file = '../config/AUTO_DRIVING.yaml'
    settings = fst.load_settings(settings_file)
    # def_host_ip = settings['wl_host']
    def_host_ip = (check_output(['hostname', '-I'])).decode('utf-8').split(' ')[0]
    print('Loaded settings file: ' + settings_file)

    parser = argparse.ArgumentParser(description='''
        Autonomous Driving System
    ''')
    parser.add_argument('--server_ip', help='server ip address', default=def_host_ip)
    args = parser.parse_args()

    # print('Opening RPI Camera')
    image_loader = RpiCamera(img_cf='jpg')
    image_loader.play_in_background()

    # Receive commands from remote client
    settings['wl_host'] = args.server_ip

    # Logging
    my_logger.info('auto_driving.main: settings file is ' + str(settings_file))
    my_logger.info('auto_driving.main: ds root is ' + str(settings['ds_root']))

    # init
    my_system = MySystem(settings, image_loader)
    # print Instructions:
    print(instructions)
    # start
    my_system.run()
    # clean
    image_loader.stop()
    image_loader.close()
    my_system.clean()
