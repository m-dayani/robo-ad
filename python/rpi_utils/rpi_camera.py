"""
    There are an old and new camera package
    Steps:
        1. enable camera: `sudo raspi-config`
        2. Test
            CL Tools:
                raspistill -o output.jpg
                raspivid -o home/pi/video.h264 -w 1024 -h 768
        3. Install necessary packages:
            pip install "picamera[array]"
        4. Run the program:
            python test_image.py

    Another option is picamzero:
        sudo apt update
        sudo apt install python3-picamzero

        rpicam-still -o ~/Desktop/image.jpg
        rpicam-still -o ~/Desktop/image-small.jpg --width 640 --height 480
        rpicam-vid -o ~/Desktop/video.mp4
        
    picamera won't work for arm64, -> use picamera2 instead:
        https://github.com/raspberrypi/picamera2

    Refs:
        - [Accessing the Raspberry Pi Camera with OpenCV and Python]
            (https://pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/)
        - [Getting started with the Camera Module]
            (https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/3)
        - [How to Take Pictures and Videos With a Raspberry Pi]
            (https://www.circuitbasics.com/introduction-to-the-raspberry-pi-camera/)
        - [Video Streaming with Raspberry Pi Camera]
            (https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/)
        - [About the Camera Modules](https://www.raspberrypi.com/documentation/accessories/camera.html)
"""

import time
import threading
import sys

import cv2
import numpy as np
from picamera2 import Picamera2
Picamera2.set_logging(Picamera2.ERROR)

sys.path.append('../')
from data_loader.image_loader import ImageLoader, ImageTs
import my_logging.my_logger as logger

my_logger = logger.setup_default_logger()


def t_picamzero():
    from picamzero import Camera

    cam = Camera() # for picamzero
    cam = PiCamera  # for picamera
    cam.flip_camera(hflip=True)
    cam.start_preview()
    # Keep the preview window open for 5 seconds
    time.sleep(5)

    cam.take_photo("~/Desktop/new_image.jpg")

    cam.capture_sequence("~/Desktop/sequence.jpg", num_images=3, interval=2)

    cam.record_video("~/Desktop/new_video.mp4", duration=5)

    cam.stop_preview()


def t_picamera():
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    class RpiCamera:
        def __init__(self, res=(640, 480), fps=30, img_cf='numpy'):
            self.resolution = res
            self.fps = fps
            self.camera = PiCamera()
            self.camera.resolution = self.resolution
            self.camera.framerate = self.fps
            self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

            self.frame = None
            self.stop_stream = False
            self.is_ok = True

            # image compression format
            self.img_cf = img_cf
            self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        def compress(self, data):
            if self.img_cf == 'jpg' or self.img_cf == 'png':
                result, data_comp = cv2.imencode('.' + self.img_cf, data, self.encode_param)
                if result:
                    data = data_comp
            return data

        def get_next(self):
            return self.compress(self.frame)

        def run(self):
            for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
                # grab the raw NumPy array representing the image, then initialize the timestamp
                # and occupied/unoccupied text
                image = frame.array
                if image is None:
                    self.is_ok = False
                    self.stop_stream = True
                    self.frame = None
                    break
                self.frame = np.copy(image)
                # clear the stream in preparation for the next frame
                self.rawCapture.truncate(0)
                # if the `q` key was pressed, break from the loop
                if self.stop_stream:
                    self.frame = None
                    break


    def t_image_capture():
        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        rawCapture = PiRGBArray(camera)
        # allow the camera to warmup
        time.sleep(0.1)
        # grab an image from the camera
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        # display the image on screen and wait for a keypress
        cv2.imshow("Image", image)
        cv2.waitKey(0)


    def t_video_stream():
        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))
        # allow the camera to warmup
        time.sleep(0.1)
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            # show the frame
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break


    def t_circuit_basics():
        camera = PiCamera()

        # To take a photo:
        camera.start_preview()
        time.sleep(5)
        camera.capture('/home/pi/Desktop/image.jpg')
        camera.stop_preview()

        # To take a video:
        camera.start_preview()
        camera.start_recording('/home/pi/Desktop/video.h264')
        time.sleep(5)
        camera.stop_recording()
        camera.stop_preview()


    def t_my_cam_stream():
        cam = RpiCamera()
        # cv_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        cv_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # fps = cam.fps
        fps = 15
        res = cam.resolution
        video = cv2.VideoWriter('output.avi', cv_fourcc, fps, res)

        cam_stream = threading.Thread(target=cam.run)
        cam_stream.start()

        start = time.time()
        cnt = 0
        duration = 10
        while time.time() - start < duration:
            frame = cam.frame
            if frame is not None:
                # print('frame ' + str(cnt) + ' received')
                video.write(frame)
                cnt += 1

        cam.stop_stream = True
        cam_stream.join()

        video.release()


class RpiCamera(ImageLoader):
    def __init__(self, res=(640, 480), fps=30, img_cf='jpg'):
        super().__init__('', '', resolution=res, fps=fps, img_cf=img_cf)

        self.camera = Picamera2()
        self.config = self.camera.create_preview_configuration({'format': 'BGR888'})
        self.camera.configure(self.config)

        self.is_ok_flag = True

        my_logger.info('rpi_utils.rpi_camera: starting RPI camera with: (res, fps, format) = (' +
                       str(res) + ', ' + str(fps) + ', BGR888)')
        self.camera.start()
        self.last_ts = -1.0

    def get_next(self):
        # self.frame = self.camera.capture_array()
        return ImageTs(self.frame, self.ts)

    def play(self):
        # self.camera.start()
        # time.sleep(2)
        p_avg = 0.0
        p_cnt = 0

        while True:
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = self.camera.capture_array()
            self.ts = time.time_ns()
            if image is None:
                self.set_ok(False)
                self.set_stop_stream(True)
                self.frame = None
                self.ts = -1.0
                break

            self.frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # if the `q` key was pressed, break from the loop
            if self.should_stop_stream():
                self.frame = None
                self.ts = -1.0
                break

            # loop time
            if self.last_ts >= 0:
                ts_diff = self.ts - self.last_ts
                p_avg += ts_diff
                p_cnt += 1
                if p_cnt % 100 == 0:
                    my_logger.info('rpi_utils.rpi_camera: Average image generation rate: ' +
                                   str(p_avg / p_cnt * 1e-6) + ' (ms)')

            self.last_ts = self.ts

    def close(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None


def t_my_cam_stream():
    cam = RpiCamera()
    # cv_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    cv_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fps = cam.fps
    fps = 15
    res = cam.resolution
    video = cv2.VideoWriter('output.avi', cv_fourcc, fps, res)

    cam_stream = threading.Thread(target=cam.run)
    cam_stream.start()

    start = time.time()
    cnt = 0
    duration = 10
    while time.time() - start < duration:
        frame = cam.frame
        if frame is not None:
            # print('frame ' + str(cnt) + ' received')
            video.write(frame)
            cnt += 1

    cam.stop_stream = True
    cam_stream.join()

    video.release()


if __name__ == "__main__":
    # t_image_capture()
    # t_video_stream()
    # t_circuit_basics()
    t_my_cam_stream()
