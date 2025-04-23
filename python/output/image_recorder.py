import os
import threading
import time

import cv2


class ImageRecorder:
    def __init__(self, ds_root, resolution=(640, 480), fps=30, fourcc='MPEG', rec_mode='video'):
        self.resolution = resolution
        self.fps = fps
        self.period = 1000.0 / (self.fps + 1e-6)    # ms
        # recording mode: 'video' or 'image'
        self.rec_mode = rec_mode
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        # self.img_provider = img_provider
        self.last_frame = None
        self.frame_lock = threading.Lock()

        self.ds_root = ds_root
        self.dir_video = os.path.join(self.ds_root, 'videos')
        if not os.path.exists(self.dir_video):
            os.mkdir(self.dir_video)
        self.dir_image = os.path.join(self.ds_root, 'images')
        if not os.path.exists(self.dir_image):
            os.mkdir(self.dir_image)

        self.recording_started = False
        self.recording_lock = threading.Lock()
        self.thread_recording = None
        self.video_idx = 0
        self.image_idx = 0

        self.last_ts = -1

        print('ImageRecorder: Initialized Successfully in ' + self.rec_mode +
              ' mode with period: ' + str(self.period) + ' ms')

    def set_last_frame(self, frame):
        self.frame_lock.acquire()
        self.last_frame = frame
        self.frame_lock.release()

    def get_last_frame(self):
        self.frame_lock.acquire()
        frame = self.last_frame
        self.frame_lock.release()
        return frame

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
            self.thread_recording = threading.Thread(target=self.run)
            self.thread_recording.start()

    def run(self):
        video = None
        if self.rec_mode == 'video':
            # video writer
            video_file = os.path.join(self.dir_video, 'video' + str(self.video_idx) + '.avi')
            self.video_idx += 1
            video = cv2.VideoWriter(video_file, self.fourcc, self.fps, self.resolution)

        while True:
            ts = time.time_ns()
            if self.last_ts < 0:
                self.last_ts = ts
            elif (ts - self.last_ts) * 1e-6 >= self.period:
                # write frames
                img_obj = self.get_last_frame()
                if img_obj is not None and img_obj.frame is not None:
                    if video is not None:
                        video.write(img_obj.frame)
                    else:
                        img_path = os.path.join(self.dir_image, str(img_obj.ts) + '.png')
                        cv2.imwrite(img_path, img_obj.frame)
                self.last_ts = ts

            if not self.is_recording():
                break

        if video is not None:
            video.release()

    def stop_recording(self):
        if self.is_recording():
            print('Stop Recording')
            self.set_recording(False)
            if self.thread_recording is not None:
                self.thread_recording.join(5)
                self.thread_recording = None

    def img_capture(self):
        frame = self.get_last_frame()
        if frame is not None:
            print('Take Picture')
            image_name = os.path.join(self.dir_image, 'image' + str(self.image_idx) + '.png')
            self.image_idx += 1
            cv2.imwrite(image_name, frame.frame)

