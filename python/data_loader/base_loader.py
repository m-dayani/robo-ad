import threading


class DataLoader:
    def __init__(self):
        self.is_ok_flag = False
        self.is_ok_lock = threading.Lock()
        self.stop_stream = False
        self.stop_stream_lock = threading.Lock()
        self.bg_thread = None
        self.data = None

    def is_ok(self):
        self.is_ok_lock.acquire()
        flag = self.is_ok_flag
        self.is_ok_lock.release()
        return flag

    def set_ok(self, flag):
        self.is_ok_lock.acquire()
        self.is_ok_flag = flag
        self.is_ok_lock.release()

    def should_stop_stream(self):
        self.stop_stream_lock.acquire()
        flag = self.stop_stream
        self.stop_stream_lock.release()
        return flag

    def set_stop_stream(self, flag):
        self.stop_stream_lock.acquire()
        self.stop_stream = flag
        self.stop_stream_lock.release()

    def get_next(self):
        # virtual method, implement
        return self.data

    def play(self):
        while self.is_ok():
            self.data = self.get_next()
            if self.should_stop_stream():
                break

    def play_in_background(self):
        if self.bg_thread is None:
            self.bg_thread = threading.Thread(target=self.play)
            self.bg_thread.start()

    def reset(self):
        self.stop()

    def stop(self):
        self.set_stop_stream(True)
        if self.bg_thread is not None:
            self.bg_thread.join(5)
            self.bg_thread = None

    def close(self):
        self.stop()



