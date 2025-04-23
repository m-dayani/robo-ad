import threading
import os
import time


class TabularTextWriter:
    def __init__(self, ds_root, get_line, freq=100):
        self.ds_root = ds_root
        self.recording_started = False
        self.recording_lock = threading.Lock()
        self.thread_recording = None

        self.dir_tt = os.path.join(ds_root, 'sensors')
        os.makedirs(self.dir_tt, exist_ok=True)
        self.get_line = get_line
        self.file_idx = 0
        self.freq = freq    # (Hz)
        self.period = 1000.0 / (freq + 1e-6)    # (ms)

        print('TabularTextWriter: initialized with period: ' + str(self.period) + ' ms')

    def is_recording(self):
        self.recording_lock.acquire()
        flag = self.recording_started
        self.recording_lock.release()
        return flag

    def set_recording(self, flag):
        self.recording_lock.acquire()
        self.recording_started = flag
        self.recording_lock.release()

    def start_recording(self):
        if not self.is_recording():
            print('Start Recording')
            self.set_recording(True)
            self.thread_recording = threading.Thread(target=self.tt_recording)
            self.thread_recording.start()

    def tt_recording(self):
        # write commands and other sensor data
        cmd_file_name = os.path.join(self.dir_tt, 'cmd' + str(self.file_idx) + '.txt')
        self.file_idx += 1
        with open(cmd_file_name, 'w') as cmd_file:
            last_ts = -1
            while True:
                ts = time.time_ns()

                if last_ts < 0:
                    last_ts = ts
                elif (ts - last_ts) * 1e-6 >= self.period:
                    # write sensor and commands
                    line = self.get_line()
                    cmd_file.writelines(line)
                    last_ts = ts

                if not self.is_recording():
                    break

    def stop_recording(self):
        if self.is_recording():
            print('Stop Recording')
            self.set_recording(False)
            if self.thread_recording is not None:
                self.thread_recording.join(5)
                self.thread_recording = None

