# Interacts with sonar sensor and gets distance to obstacles from each sensor
# ref: https://projects.raspberrypi.org/en/projects/physical-computing/12
import threading

import serial
import time

from serial.serialutil import SerialException

import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


class MyArduino:

    def __init__(self, port, bd, timeout=0.1, get_state=None, send_delay=0.1):
        self.port = port
        self.bd = bd
        self.timeout = timeout
        self.arduino = None
        self.get_state = get_state
        self.send_delay = send_delay    # (sec)
        self.log_rate = 100

        self.started = False
        self.started_lock = threading.Lock()
        self.send_thread = None

        self.init()

    def init(self):
        try:
            self.arduino = serial.Serial(port=self.port, baudrate=self.bd, timeout=self.timeout)
            time.sleep(2)
            print('Serial opened successfully, (port, bd, timeout) = ' +
                  str(self.port) + ', ' + str(self.bd) + ', ' + str(self.timeout))
        except serial.SerialException:
            print("Error: cannot open Arduino")

    def close(self):
        self.stop()
        if self.arduino is not None:
            self.arduino.close()
            print('MyArduino: Connection closed successfully')

    def transmit(self, msg):
        if self.arduino is not None:
            self.send(msg)
            time.sleep(0.05)
            msg_rx = self.arduino.readline()
            # print(msg)
            return msg_rx.strip()
        return ''

    def send(self, msg: str):
        if self.arduino is not None:
            try:
                self.arduino.write(bytes(msg+"\n", 'utf-8'))
            except SerialException:
                my_logger.debug("my_serial.arduino.MyArduino.send: Error writing to Arduino: " + msg)

    def set_started(self, flag):
        self.started_lock.acquire()
        self.started = flag
        self.started_lock.release()

    def is_started(self):
        self.started_lock.acquire()
        flag = self.started
        self.started_lock.release()
        return flag

    def stop(self):
        self.set_started(False)
        if self.send_thread is not None:
            self.send_thread.join(5)
            self.send_thread = None
        my_logger.info('MyArduino.start_in_background: BG serial transfer stopped')

    def start_in_background(self):
        if self.is_started():
            self.stop()
        self.set_started(True)
        self.send_thread = threading.Thread(target=self.run)
        self.send_thread.start()
        my_logger.info('MyArduino.start_in_background: Serial transfer started in BG with delay ' +
                       str(self.send_delay) + ' s')

    def run(self):
        cnt = 0
        while self.is_started():
            # convert and send message to the Arduino
            if self.get_state is not None:
                msg = self.get_state()
                self.send(msg)
                if cnt % self.log_rate == 0:
                    my_logger.info('MyArduino.run, sent message: ' + str(msg))
                cnt += 1
            time.sleep(self.send_delay)

    def get_distance(self, ch_id=0):
        dist = -1.0
        # if self.dist_mode == 'arduino':
        #     # request distance from remote
        #     dist_str = self.transmit('get_dist:'+str(ch_id)+":")
        #     if len(dist_str) > 0:
        #         # print(dist_str)
        #         try:
        #             dist = float(dist_str)
        #         except ValueError:
        #             print('got non-float message: ' + str(dist_str))
        # elif self.dist_mode == 'raspberry':
        #     dist = self.get_onboard_range(ch_id)
        return dist

    def get_onboard_range(self, ch_id=0):
        # if self.ultrasonic is not None:
        #     return self.ultrasonic.distance
        return -1.0


if __name__ == "__main__":

    # arduino = MyArduino('/dev/ttyACM0', 9600)
    arduino = MyArduino('/dev/ttyUSB0', 19200)

    speeds = [0, 40, 80, 120, 160, 200, 240, 255]
    angles = [60, 70, 80, 90, 100, 110, 115, 120]
    n_speeds = len(speeds)
    assert n_speeds == len(angles), 'N_speeds must match N_angles'

    for i in range(n_speeds):
        speed = speeds[i]
        angle = angles[i]
        msg = 'dir:' + str(speed) + ':' + str(angle) + ':'
        arduino.send(msg)
        time.sleep(1.0)

    for i in range(n_speeds):
        speed = speeds[n_speeds - i - 1]
        angle = angles[n_speeds - i - 1]
        msg = 'dir:' + str(speed) + ':' + str(angle) + ':'
        arduino.send(msg)
        time.sleep(1.0)
