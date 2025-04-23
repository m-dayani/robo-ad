import sys
import argparse

import numpy as np
import cv2

sys.path.append('../')
from data_loader.image_loader import ImageLoader, StereoImage


class MultiCam(ImageLoader):
    def __init__(self, path, mode, cap_mode='stereo', fourcc_code='MJPG'):
        """
        This module is for a very especial stereo camera module (OV9732) that I've worked with
        For most scenarios you should work with the image_loader tool
        :param path: image folder path, images file path, video path, stereo camera port
        :param mode: image, image_folder, video, or stream
        :param cap_mode: mono: (640, 480)-30fps, stereo: (1280, 480)-15fps
        """

        self.path = path
        self.mode = mode
        self.cap_mode = cap_mode
        self.fourcc_code = fourcc_code

        self.resolution = (1280, 480)
        self.fps = 15
        self.hw = 640
        if cap_mode == 'mono':
            self.resolution = (640, 480)
            self.fps = 30

        super().__init__(path, mode, self.resolution, self.fps, self.fourcc_code)
        self.set_ok(True)

    def get_next(self):
        frame = super().get_next()
        frame_out = None
        if self.cap_mode == 'stereo' and frame is not None:
            img = frame.frame
            frame_out = StereoImage(img[:, :self.hw], frame.ts, frame.path, frame.img_base, img[:, self.hw:])
        return frame_out


def t_stereo_stream(cam_port):
    cam = cv2.VideoCapture(cam_port)
    # cam.set(cv2.CAP_PROP_FPS, 15)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    s, original = cam.read()
    if not s:
        exit(1)
    height, width, channels = original.shape
    print(width)
    print(height)
    i = 0
    while True:
        s, original = cam.read()
        left = original[0:height, 0:int(width / 2)]
        right = original[0:height, int(width / 2):width]
        cv2.imshow('left', left)
        cv2.imshow('Right', right)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.imwrite('stereo_image' + str(i) + '.png', original)
            i += 1

    cam.release()
    cv2.destroyAllWindows()


def t_multi_cam(path, mode, cap_mode, fourcc_code='MJPG'):
    cam = MultiCam(path, mode, cap_mode, fourcc_code)
    wait = int(1000 / cam.fps)

    while cam.is_ok:
        img0, img1 = cam.get_next()

        if img0 is not None:
            cv2.imshow('Left', img0)
        if img1 is not None:
            cv2.imshow('Right', img1)

        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q'):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        This module is for a especial stereo camera module (OV9732)
    """)
    parser.add_argument('path', help='sequence path, image folder, video port, image file')
    parser.add_argument('--mode', help='image, image_folder, video, or stream', default='video')
    args = parser.parse_args()

    # t_stereo_stream(2)
    path = args.path
    mode = args.mode
    t_multi_cam(path, mode, 'stereo')

