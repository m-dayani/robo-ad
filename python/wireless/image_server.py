#!/usr/bin/env python

import argparse
import select
import socket
import sys
import threading
import time

import numpy as np

sys.path.append('../')
sys.path.append('../rpi_utils/')
sys.path.append('../data_loader/')
from wl_socks import MyWirelessDriver, Server
# from tools.utils import MyTimer
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()
try:
    from rpi_utils.rpi_camera import RpiCamera
    test_environment = False
except (ImportError, RuntimeError):
    from data_loader.image_loader import ImageLoader
    test_environment = True


class ImageServer(Server):

    def __init__(self, host, port, send_mode=0, fps=30, show_time=False):
        super().__init__(host, port, show_time=show_time)
        # self.image_loader = image_provider
        # send_mode: 0: async, 1: sync 1 ack, 2: sync double ack
        self.send_mode = send_mode
        self.last_frame = None
        self.last_img_ts = -1
        self.frame_lock = threading.Lock()
        self.fps = fps
        self.ftime = 1000.0 / (fps + 1e-6)  # (ms)

    def set_last_frame(self, frame):
        self.frame_lock.acquire()
        self.last_frame = frame
        self.frame_lock.release()

    def get_last_frame(self):
        self.frame_lock.acquire()
        frame = self.last_frame
        self.frame_lock.release()
        return frame

    def server_trans(self):
        # data retrieval
        img_ts = self.get_last_frame()
        if img_ts is None:
            return True

        # data manipulation and compression
        # this conversion logic must be implemented at the provider level

        # data serialization
        frame = img_ts.frame
        if frame is None:
            return True

        # timestamp check: high input frame rates can congest weak networks
        ts = img_ts.ts
        if self.last_img_ts >= 0:
            t_diff_ms = (ts - self.last_img_ts) * 1e-6
            if t_diff_ms < self.ftime:
                # print('ImageServer.server_trans: high frame rate detected')
                return True
        self.last_img_ts = ts

        data_bytes = frame.tobytes()
        n_bytes = len(data_bytes)
        # print('sending: ' + str(n_bytes) + ' bytes of image at ' + str(img_ts.ts))
        payload = MyWirelessDriver.array_to_str(np.concatenate(([n_bytes], frame.shape, [ts])), 0)

        try:
            self.send(self.client_socket, MyWirelessDriver.encode('data', payload))
        except ConnectionResetError:
            print('ImageServer.server_trans: Connection reset by peer')
            return False

        if self.send_mode == 0:
            # async operation
            self.send(self.client_socket, data_bytes)
        else:
            # sync operation
            res, data = self.receive(self.client_socket, -1, self.MSG_LEN)
            msg_str = MyWirelessDriver.decode_str(data)
            if 'ack' in msg_str:
                self.send(self.client_socket, data_bytes)

                if self.send_mode == 2:
                    res, data = self.receive(self.client_socket, -1, self.MSG_LEN)
                    msg_str = MyWirelessDriver.decode_str(data)
                    # if 'ack' in msg_str:
                    #     print('remote received image message')
            else:
                print('Remote controller cannot acknowledge image header, aborting stream')
                return False

        self.timer.roll(time.time_ns(), 'Average send image loop time: ')
        if self.timer.t_cnt % self.timer.show_rate == 0:
            my_logger.info('ImageServer.server_trans, Average send image loop time: ' +
                           str(self.timer.t_avg * 1e-6) + ' (ms)')

        return True


def t_initial():
    imgcounter = 1
    basename = "image%s.png"

    HOST = '127.0.1.1'
    PORT = 6666

    connected_clients_sockets = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)

    connected_clients_sockets.append(server_socket)

    while True:

        read_sockets, write_sockets, error_sockets = select.select(connected_clients_sockets, [], [])

        for sock in read_sockets:

            if sock == server_socket:

                sockfd, client_address = server_socket.accept()
                connected_clients_sockets.append(sockfd)

            else:
                try:

                    data = sock.recv(4096)
                    txt = str(data)

                    if data:

                        if data.startswith('SIZE'):
                            tmp = txt.split()
                            size = int(tmp[1])

                            print('got size')

                            sock.sendall("GOT SIZE")

                        elif data.startswith('BYE'):
                            sock.shutdown()

                        else :

                            myfile = open(basename % imgcounter, 'wb')
                            myfile.write(data)

                            data = sock.recv(40960000)
                            if not data:
                                myfile.close()
                                break
                            myfile.write(data)
                            myfile.close()

                            sock.sendall("GOT IMAGE")
                            sock.shutdown()
                except:
                    sock.close()
                    connected_clients_sockets.remove(sock)
                    continue
            imgcounter += 1
    server_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Emulate a remote robot server (like RPI or ESP32-cam). 
        ''')
    parser.add_argument('--server_ip', help='server ip address', default='127.0.1.1')
    parser.add_argument('--server_port_data', help='server port for video stream', default=8080)
    args = parser.parse_args()

    host = args.server_ip
    port_data = int(args.server_port_data)

    # for dev environment, use webcam, in RPI use RPI camera
    if test_environment:
        print('Test Environment: Opening test video loader (default webcam)')
        image_provider = ImageLoader('0', 'stream', fps=30)
    else:
        print('Opening RPI Camera')
        image_provider = RpiCamera(img_cf='jpg')

    video_server = ImageServer(host, port_data, image_provider, show_time=True)

    # you need to play RPI camera data provider
    if not test_environment:
        image_provider.play_in_background()

    # run main server thread
    video_server.run()
    video_server.close()
