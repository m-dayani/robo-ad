#!/usr/bin/env python
# TCP client - transmit a float in binary every 3 seconds to ESP32 server
# From: https://forum.arduino.cc/t/setup-communication-between-desktop-pc-and-esp32-over-wifi/1152476/9
import argparse
import time
import threading
import socket
import struct
import requests

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from wl_socks import MyWirelessDriver, Client, MySocket


class HttpStreamListener(object):
    def __init__(self, host, port=80, show_frames=False):

        self.initial_timeout = 5
        self.receive_timeout = 5
        self.max_recv_bytes = 100000000
        self.seg_len = 1024
        self.url = 'http://' + host + ':' + str(port) + '/'

        self.stream = None
        self.frame = None
        self.stop_stream = False
        self.show_frames = show_frames

    def run(self):
        self.stream = requests.get(self.url, stream=True, timeout=self.initial_timeout)
        self.stream.raise_for_status()

        # Test connection
        content_len = self.stream.headers.get('Content-Length')
        if content_len is not None and int(content_len) > self.max_recv_bytes:
            raise ValueError('response too large')

        size = 0
        start = time.time()
        waiting_for_frame = False
        msg_len = 0
        frame_bytes = b''
        bytes_recv = 0

        for chunk in self.stream.iter_content(self.seg_len):
            # if time.time() - start > receive_timeout:
            #     raise ValueError('timeout reached')

            # size += len(chunk)
            # if size > your_maximum:
            #     raise ValueError('response too large')

            if self.stop_stream:
                break

            # do something with chunk
            if waiting_for_frame:
                frame_bytes += chunk
                bytes_recv += len(chunk)
                if bytes_recv >= msg_len:
                    # do sth with frame
                    waiting_for_frame = False

                    # Load image from BytesIO
                    im = Image.open(BytesIO(frame_bytes))
                    self.frame = np.array(im)

                    if self.show_frames:
                        cv2.imshow('ESP32 Image', self.frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.stop_stream = True
                            break
            else:
                header = chunk.decode('utf-8')
                if 'Content' in header:
                    lines = header.splitlines()
                    for line in lines:
                        if len(line) <= 0:
                            continue
                        if 'Content-Length' in line:
                            msg_len = int(line.split(': ')[-1])
                            # print('receiving new frame with length: ' + str(msg_len))
                            waiting_for_frame = True
                            frame_bytes = b''
                            bytes_recv = 0


class ImageClient(Client):

    def __init__(self, host, port_data, stream_mode=0, show_time=False):
        super().__init__(host, port_data, show_time=show_time)

        self.frame = None

        # stream mode: 0: async no-ack, 1: sync 1 ack, 2: sync 2 ack, 3: esp32,
        self.stream_mode = stream_mode
        self.esp32_stream = HttpStreamListener(host, port_data, True)
        self.raw_data = b''
        self.should_stop = False
        self.data_label_bytes = 'data:'.encode('utf-8')

    def run(self):
        if self.stream_mode == 3:
            self.esp32_stream.run()
        else:
            while True:
                # fetch images and sensor data
                res, data = self.receive_c(-1)
                # print(len(data))
                if res:
                    self.process_stream(data)

                if self.should_stop:
                    break

    def img_recv_ack(self, data):
        # single acknowledge image receiver may throw exception,
        # use double-ack version instead
        # res = super().server_trans()
        # if res:
        self.raw_data = data
        msg_str = MyWirelessDriver.decode_str(self.raw_data)
        if 'data' in msg_str:
            msg_parts = MyWirelessDriver.decode(self.raw_data)[0][0]
            payload = np.int32(msg_parts[1].split(','))
            self.send_c(MyWirelessDriver.encode('ack'))
            res, self.raw_data = self.receive_c(payload[0])
            self.process_img_bytes(self.raw_data, payload[1:])
        # return res

    def img_recv_ack2(self, data):
        if 'EOF' in MyWirelessDriver.decode_str(data):
            return
        cmd_data = MyWirelessDriver.decode(data)[0][0]
        if len(cmd_data) >= 2:
            # cmd = cmd_data[0]
            data = cmd_data[1]
            data_parts = np.int32(data.split(','))
            self.send_c(MyWirelessDriver.encode('ack'))
            res, frame_data = self.receive_c(data_parts[0])
            if res:
                self.process_img_bytes(frame_data, data_parts[1:])
                if self.stream_mode == 1:
                    self.send_c(MyWirelessDriver.encode('ack'))
                # if self.show_frames:
                #     cv2.imshow('Received Image', self.frame)
                #     cv2.waitKey(1)
            else:
                return

        # self.last_t_stream = self.show_loop_time('Stream client loop time: ', self.last_t_stream)
        # cv2.destroyAllWindows()

    def img_recv_no_ack(self, data):

        self.raw_data += data
        start_idx = self.raw_data.find(self.data_label_bytes)
        while start_idx >= 0:
            end_idx = self.raw_data.find(';data'.encode('utf-8'))
            if end_idx < 0:
                break
            end_idx += 1
            header_bytes = self.raw_data[start_idx:end_idx]
            label, payload = MyWirelessDriver.decode(header_bytes)[0][0][:2]
            size_info = np.int32(payload.split(','))
            n_bytes = size_info[0]
            bytes_end = end_idx + n_bytes
            if len(self.raw_data) < bytes_end:
                break

            image_bytes = self.raw_data[end_idx:bytes_end]
            self.process_img_bytes(image_bytes, size_info[1:])

            self.raw_data = self.raw_data[bytes_end:]
            start_idx = self.raw_data.find(self.data_label_bytes)

            # self.last_t_stream = self.show_loop_time('Stream client loop time: ', self.last_t_stream)

            # if self.show_frames:
            #     cv2.imshow('Received Image', self.frame)
            #     cv2.waitKey(1)

        # cv2.destroyAllWindows()

    def img_recv_no_ack2(self, data):

        self.raw_data += data
        idx = self.raw_data.rfind('data:'.encode('utf-8'))

        if idx != 0:
            idx_end = self.raw_data.find(';'.encode('utf-8'))

            img_info = self.raw_data[:idx_end]
            img_info_str = img_info.decode('utf-8')

            img_ts = np.int64(img_info_str.split(',')[-1])
            self.timer.roll(img_ts, msg='Average rec image period: ')

            img_data = self.raw_data[(idx_end + 1):(idx_end + idx + 1)]
            recv_image = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(recv_image, cv2.IMREAD_UNCHANGED)
            self.raw_data = self.raw_data[idx:]

            if frame is not None:
                cv2.imshow('Image', frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    self.should_stop = True

    def process_stream(self, data):
        if self.stream_mode == 0:
            self.img_recv_no_ack2(data)
        elif self.stream_mode == 2:
            self.img_recv_ack2(data)
        else:
            self.img_recv_ack(data)

    def process_img_bytes(self, img_bytes, img_size):
        recv_image = np.frombuffer(img_bytes, np.uint8)
        self.frame = recv_image.reshape(img_size)

def t_esp32_pc_recv():
    TCP_IP = '192.168.124.100'
    TCP_PORT = 10000
    BUFFER_SIZE = 1024
    value = 3.145

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    while True:
        ba = bytearray(struct.pack("f", value))
        s.send(ba)
        value=value+5.0
        time.sleep(3)
    s.close()


def t_better_trans(my_server, my_client, frame):
    # image transmission works but the recipient must now data size and other info

    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()

    # if ret:
    time.sleep(1)
    my_client.connect()
    time.sleep(1)
    # You can also use pickle to support general structures
    frame_bytes = frame.tobytes()
    data_arr = np.concatenate(([len(frame_bytes)], frame.shape))
    payload = MyWirelessDriver.array_to_str(data_arr, 0)
    data_header = MyWirelessDriver.encode('data', payload)
    my_client.send_c(data_header)
    time.sleep(0.01)
    # server should send an ack, otherwise it'll get streamed data!
    res, recv_msg = my_client.receive_c(-1)
    if 'ack' in MyWirelessDriver.decode_str(recv_msg):
        my_client.send_c(frame_bytes)
        time.sleep(1)
        recv_image = np.frombuffer(my_server.raw_data, np.uint8)
        recv_image = recv_image.reshape(frame.shape)
        # cv2.imshow('Received Image', recv_image)
        # cv2.waitKey()

    # cap.release()
    # cv2.destroyAllWindows()


def t_esp32_other():
    esp32_image_stream = HttpStreamListener('192.168.219.100', 8080)
    thread_stream = threading.Thread(target=esp32_image_stream.run)
    thread_stream.start()

    # while True:
    #     frame = esp32_image_stream.frame
    #     if frame is not None:
    #         cv2.imshow('ESP32 Image', frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             break

    esp32_image_stream.stop_stream = True
    thread_stream.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
        Wireless image client test
    ''')
    parser.add_argument('--host_ip', help='host ip', default='127.0.1.1')
    parser.add_argument('--port', help='data port', default=8080)
    args = parser.parse_args()

    host = args.host_ip
    port = int(args.port)
    my_client = ImageClient(host, port, show_time=True)
    my_client.connect()
    my_client.run()

    my_client.close()




