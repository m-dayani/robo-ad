"""
    Autonomous Driving System (Recording)

    This is much like the AD System but doesn't include Auto Driving and Ultrasonic submodules
    Used for recording commands
"""

import argparse
import datetime
import os
import sys
import threading
import time
import signal
import copy
from subprocess import check_output

import numpy as np

sys.path.append('../')
sys.path.append('../wireless/')
sys.path.append('../rpi_utils/')
sys.path.append('../data_loader/')
from my_serial.arduino import MyArduino
from wireless.image_server import ImageServer
from output.image_recorder import ImageRecorder
from output.tt_writer import TabularTextWriter
import tools.utils as fst
from rpi_utils.rpi_camera import RpiCamera
from main import CmdReceiver
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


instructions = '''
================== Recording System Instructions ==================
Q_KEY:              Tack Pictures
Num_KEY 1:          ON: Recording, OFF: Stop Recording
UP/DOWN Arrow:      Trottle in manual control
LEFT/RIGHT Arrow:   Steering in manual control
ESC:                Stop and quit
===================================================================
'''


class RecordingSystem(object):
    def __init__(self, settings, image_loader, use_video_server=False, video_cap_mode='video'):

        self.image_loader = image_loader

        # Arduino: serial transmission between RPI server and microcontroller
        self.arduino = MyArduino(settings['serial_port'], settings['serial_bd'],
                                 get_state=self.serialize_out_state,
                                 send_delay=settings['serial_send_delay'])

        # CMD Server
        host = settings['wl_host']
        port = int(settings['wl_port'])
        self.cmd_server = CmdReceiver(host, port)

        # Video server slows down the pipeline, so it's ignored
        # self.use_video_server = use_video_server
        self.video_server = None
        if use_video_server:
            port_data = int(settings['wl_port_data'])
            self.video_server = ImageServer(host, port_data, settings['stream_mode'])

        self.client_connected = False
        self.thread_ctrl_server = None
        self.thread_data_server = None

        # make dataset dirs
        record_root = os.path.join(settings['ds_root'], 'datasets',
                                   datetime.datetime.now().strftime('%Y-%b-%d-%H-%M-%S'))
        try:
            os.makedirs(record_root, exist_ok=True)
        except PermissionError:
            print('Permission error while creating root dataset: ' + record_root)
            record_root = '.'

        # set the environment variable to the last record_root
        os.environ['DS_ROOT'] = record_root

        self.v_vel_default = np.zeros(4, dtype=np.float32)
        self.v_vel_default[:2] = 127.5
        self.v_vel = np.copy(self.v_vel_default)
        self.v_pb = np.zeros(4, dtype=np.bool_)
        self.v_sw = np.zeros(4, dtype=np.bool_)
        # out state: [f/b (trot), l/r (steering), obj, RES]
        self.out_state = np.uint8(self.v_vel)

        self.lim_speed = [53, 203]
        self.lim_angle = [60, 120]

        self.video_recorder = ImageRecorder(record_root, fps=15, rec_mode=video_cap_mode)
        self.tt_writer = TabularTextWriter(record_root, self.get_line)
        self.recording_started = False

        self.sig_int = False

        # logging
        my_logger.info('auto_driving.main: server_ip is ' + str(host))
        my_logger.info('auto_driving.main: created a dataset at ' + str(record_root))

    def run(self):
        signal.signal(signal.SIGINT, self.signal_int_handler)

        # listen for connections
        self.thread_ctrl_server = threading.Thread(target=self.cmd_server.run)
        self.thread_ctrl_server.start()
        if self.video_server is not None:
            self.thread_data_server = threading.Thread(target=self.video_server.run)
            self.thread_data_server.start()

        print('Waiting for client to connect ...')
        while self.cmd_server.client_socket is None:
            if self.sig_int:
                return
            time.sleep(1)

        self.arduino.start_in_background()

        # this flag is for reconnecting other clients, but it's not available now
        self.client_connected = True

        while True:
            # if any client is connected, start the main process
            # check cmd commands (stop, run, auto_driving, ...)
            self.check_control()

            if not self.client_connected:
                print('AD_System.run: Client disconnected, breaking the main loop')
                break

            # get last image from image loader
            frame = self.image_loader.get_next()
            if frame is not None:
                # supply input image to the other components
                # (image server, image recorder, and auto-driving/ml-detector)
                # for the last two, nothing happens if these are not started,
                # but in case of video_recorder, setting frame is necessary for single image capture
                if self.video_server is not None:
                    compressed_frame = copy.deepcopy(frame)
                    compressed_frame.frame = self.image_loader.compress(compressed_frame.frame)
                    self.video_server.set_last_frame(compressed_frame)

                self.video_recorder.set_last_frame(frame)

            # decide what command to send to the arduino or record (from 3 sources)
            self.update_control()

        if self.recording_started:
            self.tt_writer.stop_recording()
            self.video_recorder.stop_recording()
            self.recording_started = False

        self.arduino.stop()

    def clean(self):

        if self.recording_started:
            self.tt_writer.stop_recording()
            self.video_recorder.stop_recording()
            self.recording_started = False

        self.arduino.close()

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

        # record videos
        if self.v_sw[0]:
            # checking internal flags is much faster than synced writers' flag
            if not self.recording_started:
                self.video_recorder.start_recording()
                self.tt_writer.start_recording()
                self.recording_started = True
        else:
            if self.recording_started:
                self.video_recorder.stop_recording()
                self.tt_writer.stop_recording()
                self.recording_started = False

        # take pictures
        if self.v_pb[0]:
            self.video_recorder.img_capture()

    def update_control(self):
        self.out_state = self.get_man_state()

    def get_man_state(self):
        # convert the internal state to the serial state
        state = np.uint8(self.v_vel)
        state[2:] = 0
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
        return str(int(time.time_ns())) + ',' + ','.join(map(str, self.out_state)) + '\r\n'

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
    parser.add_argument('--video_cap_mode', help='video capture mode (default: video)', default='video')
    parser.add_argument('--use_video_server', help='use video server? (default: False)', default=0)
    args = parser.parse_args()

    # if test environment, run video player, else: run RPI camera
    # print('Opening RPI Camera')
    image_loader = RpiCamera(img_cf='jpg')
    image_loader.play_in_background()

    # Receive commands from remote client
    settings['wl_host'] = args.server_ip

    # Logging
    my_logger.info('auto_driving.main: settings file is ' + str(settings_file))
    my_logger.info('auto_driving.main: video_cap_mode is ' + str(args.video_cap_mode))
    my_logger.info('auto_driving.main: ds root is ' + str(settings['ds_root']))

    use_vs = int(args.use_video_server) == 1

    # init
    my_system = RecordingSystem(settings, image_loader, use_vs, args.video_cap_mode)
    # print instructions
    print(instructions)
    # start
    my_system.run()
    # clean
    image_loader.stop()
    image_loader.close()
    my_system.clean()
