import time

import numpy as np
import pickle
import yaml


class MyTimer(object):
    def __init__(self, show_time=False):
        self.t0 = -1
        self.t1 = -1
        self.t_avg = 0.0
        self.t_cnt = 0
        self.show_time = show_time
        self.show_rate = 100

    def roll(self, ts, msg='Average time difference: '):
        self.t1 = ts
        # self.t1 = time.time_ns()
        if self.t0 >= 0:
            t_diff = self.t1 - self.t0
            self.t_avg = (t_diff + self.t_cnt * self.t_avg) / (self.t_cnt + 1)
            self.t_cnt += 1
            self.print(msg=msg)
        self.t0 = self.t1
            
    def print(self, msg='Average time difference: '):
        if self.show_time:
            if self.t_cnt % self.show_rate == 0:
                print(msg + str(self.t_avg * 1e-6) + ' (ms)')


def load_settings(file_name):
    with open(file_name) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return None


if __name__ == "__main__":
    # Here's an example dict
    grades = {'Alice': 89, 'Bob': 72, 'Charles': 87}

    # Use dumps to convert the object to a serialized string
    serial_grades = pickle.dumps(grades)

    # Use loads to de-serialize an object
    received_grades = pickle.loads(serial_grades)

    # Converting NumPy array to byte format
    byte_output = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).tobytes()

    # Converting byte format back to NumPy array
    array_format = np.frombuffer(byte_output)
