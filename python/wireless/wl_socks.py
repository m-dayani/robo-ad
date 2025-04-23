"""
    A little delay between operations makes program more stable
    This package is primarily developed for Desktop-RPI communication
    Notes:
        - If you don't want to get a bulk of streamed data,
            implement a send-ack-respond mechanism.

    refs:
        https://www.geeksforgeeks.org/socket-programming-python/
        https://docs.python.org/3/howto/sockets.html
"""

import socket
import threading
import time
import sys

sys.path.append('../')
from tools.utils import MyTimer


class MyWirelessDriver:
    def __init__(self):
        print('MyWirelessDriver')

    @staticmethod
    def array_to_str(payload, precision=2, data_sep=','):
        if precision > 0:
            precision_str = '%.' + str(precision) + 'f'
        else:
            precision_str = '%d'
        return data_sep.join(precision_str % v for v in payload)

    @staticmethod
    def array_to_str_v(payload, precision=(2,), data_sep=','):
        n_arr = len(payload)
        assert n_arr == len(precision), 'Length of tuple of input arrays mismatch the precision array'
        msg = ''
        for i in range(n_arr):
            msg += MyWirelessDriver.array_to_str(payload[i], precision[i], data_sep)
            if i != n_arr - 1:
                msg += data_sep
        return msg

    @staticmethod
    def encode(label, payload=None, label_sep=':', data_end=';'):
        """
        :param label: string, data label
        :param payload: string (convert arrays using the methods above)
        :param label_sep:
        :param data_end:
        :return: general form: 'label[:1,2,3,...];'
        """
        if payload is None:
            msg = label
        else:
            msg = label + label_sep + payload
        return bytes(msg + data_end, 'utf-8')

    @staticmethod
    def decode_str(msg_bytes):
        if len(msg_bytes) > 0:
            return msg_bytes.decode('utf-8', errors='ignore')
        return ''

    @staticmethod
    def decode(msg_bytes, label_sep=':', data_end=';'):
        msg_list = []
        res = None
        msg_str = MyWirelessDriver.decode_str(msg_bytes)
        if len(msg_str) > 0:
            data_fields = msg_str.split(data_end)
            for field in data_fields:
                if len(field) <= 0:
                    continue
                msg_list.append(field.split(label_sep))
            if msg_str[-1] != data_end:
                res = data_fields[-1]
        return msg_list, res


class MySocket:
    def __init__(self, host, port, show_time=False):
        self.host = host
        self.port = port
        self.MSG_LEN = 1024
        self.sock = None
        self.timer = MyTimer(show_time=show_time)

        self.open()

    def open(self):
        if self.sock is None:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print("Socket successfully created")
            except socket.error as err:
                print("Socket creation failed with error %s" % err)
                self.sock = None

    @staticmethod
    def send(sock, msg_bytes):
        if sock is not None:
            try:
                sock.sendall(msg_bytes)
            except BrokenPipeError as error:
                print(error)

    @staticmethod
    def receive(sock, msg_len, chunk_len):
        recv_length = 0
        data_bytes = b''
        recv_ok = True
        if sock is not None:
            try:
                if msg_len <= 0:
                    return True, sock.recv(chunk_len)
                while recv_length < msg_len:
                    curr_bytes = sock.recv(chunk_len)
                    data_bytes += curr_bytes
                    recv_length += len(curr_bytes)
                recv_ok = recv_length == msg_len
            except OSError:
                recv_ok = False
                print('MySocket.receive: Error receiving data from socket')
        return recv_ok, data_bytes

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class Client(MySocket):
    def __init__(self, host, port, show_time=False):
        super().__init__(host, port, show_time=show_time)

    def connect(self):
        if self.sock is not None:
            try:
                self.sock.connect((self.host, self.port))
            except OSError as error:
                print(error)

    def send_c(self, msg_bytes):
        MySocket.send(self.sock, msg_bytes)

    def receive_c(self, msg_len):
        return MySocket.receive(self.sock, msg_len, self.MSG_LEN)


class Server(MySocket):
    def __init__(self, host, port, show_time=False):
        super().__init__(host, port, show_time=show_time)

        self.started = True
        self.started_lock = threading.Lock()
        self.client_socket = None
        self.client_address = None
        self.thread_accept = None

        # no assumption on the nature of data bytes
        self.raw_data = b''

        if host is None:
            try:
                self.host = socket.gethostbyname(socket.gethostname())
                print("Server host: " + str(self.host))
            except socket.gaierror:
                # this means could not resolve the host
                print("there was an error resolving the host")
                self.host = None
                # sys.exit()

    def my_listen(self):
        if self.sock is not None:
            try:
                if self.host is None:
                    self.host = socket.gethostname()
                print('Server, listening on: ' + self.host + ":" + str(self.port))
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((self.host, self.port))
                self.sock.listen(5)
            except OSError as error:
                print(error)

    def my_accept(self):
        if self.sock is not None:
            try:
                self.my_listen()
                self.client_socket, self.client_address = self.sock.accept()
                print('Connected by ' + str(self.client_address))
            except OSError as error:
                print(error)
            except AttributeError as attr_err:
                print(attr_err)
            # self.thread_accept = None

    def set_started(self, state):
        self.started_lock.acquire()
        self.started = state
        self.started_lock.release()

    def is_started(self):
        self.started_lock.acquire()
        state = self.started
        self.started_lock.release()
        return state

    def server_trans(self):
        res, self.raw_data = MySocket.receive(self.client_socket, -1, self.MSG_LEN)
        if len(self.raw_data) > 0:
            return True
        return False

    def run(self):
        self.set_started(True)
        while self.is_started():
            if self.client_socket is not None:
                res = self.server_trans()
                if not res:
                    self.stop()
                    break
            else:
                # accept in separate thread
                if self.thread_accept is None:
                    self.thread_accept = threading.Thread(target=self.my_accept, args=())
                    self.thread_accept.start()
                # self.my_accept()

    def stop(self):
        self.set_started(False)
        if self.client_socket is not None:
            self.client_socket.close()
            self.client_socket = None

    def close(self):
        super().close()
        if self.thread_accept is not None and self.thread_accept.is_alive():
            try:
                # stop accept
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.port))
                self.thread_accept.join()
                self.thread_accept = None
            except ConnectionResetError:
                print('Connection refused while canceling server accept')


def t_trans_string(my_server, my_client):
    time.sleep(1)
    my_client.connect()
    time.sleep(1)
    msg_bytes = MyWirelessDriver.encode("hello, " + host)
    my_client.send_c(msg_bytes)
    time.sleep(1)
    msg_str = MyWirelessDriver.decode_str(my_server.raw_data)
    print('Client says: ' + msg_str)


if __name__ == "__main__":

    host = '127.0.1.1'
    port = 29456
    my_server = Server(host, port)
    my_client = Client(host, port)

    th_server = threading.Thread(target=my_server.run, args=())
    th_server.start()

    t_trans_string(my_server, my_client)
    # t_trans_image(my_server, my_client)

    my_client.close()
    my_server.set_started(False)
    my_server.close()

    th_server.join()

