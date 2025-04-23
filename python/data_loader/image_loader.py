"""
    Load images from offline image folders, image files, or video files
    Use camera for live streams
"""
import glob
import os
import threading
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from base_loader import DataLoader
from tt_loader import TabularTextLoader


class ImageTs(object):
    def __init__(self, frame=None, ts=-1.0, path='', base_dir=''):
        self.frame = frame
        self.ts = ts
        self.path = path
        self.img_base = base_dir


class StereoImage(ImageTs):
    def __init__(self, frame=None, ts=-1.0, path='', base_dir='', frame1=None):
        super().__init__(frame, ts, path, base_dir)
        self.frame1 = frame1


class ImageLoader(DataLoader):
    def __init__(self, path, mode, resolution=(640, 480), fps=15, fourcc_code='MJPG', img_cf='jpg'):
        """
        Args:
            path: path to video or image file or image folder
            mode: possible modes: 'images_file', 'image', 'video', 'image_folder', 'stream'
        """
        super().__init__()
        # acceptable image suffixes
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        # acceptable video suffixes
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

        self.path = path
        self.mode = mode
        self.resolution = resolution
        self.fourcc_code = fourcc_code
        self.fps = fps

        self.ts_arr = []
        self.images = []
        self.img_base = ''
        self.image_idx = 0
        self.cap = None
        self.frame = None
        self.ts = -1.0

        # image compression format
        self.img_cf = img_cf
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        if mode == 'image':
            if self.is_image(path):
                self.images.append(path)
        elif mode == 'image_folder':
            files = glob.glob(os.path.join(path, '*'))
            files = sorted(files)
            for file_path in files:
                if self.is_image(file_path):
                    self.images.append(file_path)
        elif mode == 'video':
            self.cap = cv2.VideoCapture(path)
        elif mode == 'stream':
            self.cap = cv2.VideoCapture(int(path))
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_code))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        elif mode == 'images_file':
            images_ds = TabularTextLoader(path)
            self.ts_arr = images_ds.get_cols(0, np.float64)
            self.images = images_ds.get_cols(1)
            self.img_base = os.path.split(path)[0]

        if len(self.images) > 0 or self.cap is not None:
            self.is_ok_flag = True

    def is_image(self, img_path):
        if os.path.isfile(img_path):
            img_ext = os.path.splitext(img_path)[-1][1:]
            if img_ext in self.img_formats:
                return True
        return False

    def compress(self, data):
        if data is not None:
            if self.img_cf == 'jpg' or self.img_cf == 'png':
                result, data_comp = cv2.imencode('.' + self.img_cf, data, self.encode_param)
                if result:
                    data = data_comp
        return data

    def get_next_compressed(self):
        img_ts = self.get_next()
        if img_ts is not None:
            img_ts.frame = self.compress(img_ts.frame)
        return img_ts

    def get_num_images(self):
        return len(self.images)

    def get_image_item(self, i):
        n_images = len(self.images)
        if 0 <= i < n_images:
            image_file = os.path.join(self.img_base, self.images[i])
            frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            ts = -1.0
            if len(self.ts_arr) == n_images:
                ts = self.ts_arr[i]
            return ImageTs(frame, ts, str(image_file), self.img_base)
        return None

    def get_image_image(self):
        if len(self.images) > 0:
            img_path = self.images[0]
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            return ImageTs(frame, path=img_path)
        return None

    def get_image_image_folder(self):
        n_images = len(self.images)
        if n_images > 0 and self.image_idx < n_images:
            img_path = self.images[self.image_idx]
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.image_idx += 1
            return ImageTs(frame, path=img_path)
        return None

    def get_image_images_file(self):
        n_images = len(self.images)
        if n_images > 0 and self.image_idx < n_images:
            image_file = os.path.join(self.img_base, self.images[self.image_idx])
            frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            ts = self.ts_arr[self.image_idx]
            self.image_idx += 1
            return ImageTs(frame, ts, str(image_file), self.img_base)
        return None

    def get_image_video(self):
        if self.cap is not None:
            ret, img = self.cap.read()
            if ret:
                return ImageTs(img, time.time_ns())
        return None

    def get_next(self):
        image = None
        if self.mode == 'image':
            image = self.get_image_image()
        elif self.mode == 'image_folder':
            image = self.get_image_image_folder()
        elif self.mode == 'images_file':
            image = self.get_image_images_file()
        elif self.mode in ['video', 'stream']:
            image = self.get_image_video()

        if image is None:
            self.set_ok(False)

        return image

    def play(self):
        while self.is_ok():
            self.frame = self.get_next()
            if self.should_stop_stream():
                break

    def play_in_background(self):
        if self.bg_thread is None:
            self.bg_thread = threading.Thread(target=self.play)
            self.bg_thread.start()

    def get_test_images(self, num_test_images):
        if 0 < num_test_images < len(self.images):
            # Randomly select test images
            return random.sample(self.images, num_test_images)
        return []

    def reset(self):
        super().reset()
        self.image_idx = 0
        self.frame = None

    def close(self):
        super().close()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ImageLoaderEuRoC(ImageLoader):
    def __init__(self, path):
        super().__init__(path, 'images_file')

    def get_image_images_file(self):
        n_images = len(self.images)
        if n_images > 0 and self.image_idx < n_images:
            img_base = os.path.join(self.img_base, 'data')
            image_file = os.path.join(img_base, self.images[self.image_idx])
            frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            ts = self.ts_arr[self.image_idx]
            self.image_idx += 1
            return ImageTs(frame, ts, str(image_file), img_base)
        return None

    def get_image_item(self, i):
        n_images = len(self.images)
        if 0 <= i < n_images:
            img_base = os.path.join(self.img_base, 'data')
            image_file = os.path.join(img_base, self.images[i])
            frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            ts = -1.0
            if len(self.ts_arr) == n_images:
                ts = self.ts_arr[i]
            return ImageTs(frame, ts, str(image_file), img_base)
        return None


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        # todo: merge this to the above

        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in self.img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in self.vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.cap = None
        if any(videos):
            self.new_video(videos[0])  # new video

        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {self.img_formats}\nvideos: {self.vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = LoadImages.dataset_letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

    @staticmethod
    def letterbox(img):
        new_shape = (640, 640)
        color = (114, 114, 114)
        auto = True
        scaleFill = False
        scaleup = True
        stride = 32

        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    @staticmethod
    def dataset_letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                          stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


def t_image_loader(ds_root):
    sample_image = os.path.join(ds_root, 'images', 'image0.png')
    image_folder = os.path.join(ds_root, 'images')
    video_file = os.path.join(ds_root, 'videos', 'video9.avi')

    image_loader = ImageLoader(sample_image, 'image')
    dir_loader = ImageLoader(image_folder, 'image_folder')
    video_loader = ImageLoader(video_file, 'video')

    while image_loader.is_ok:
        image = image_loader.get_next()
        if image is not None:
            cv2.imshow('Image Loader', image)
            cv2.waitKey()

    while dir_loader.is_ok:
        image = dir_loader.get_next()
        if image is not None:
            cv2.imshow('Directory Loader', image)
            key = cv2.waitKey()
            if key & 0xFF == ord('q'):
                break

    while video_loader.is_ok:
        image = video_loader.get_next()
        if image is not None:
            cv2.imshow('Video Loader', image)
            key = cv2.waitKey(30)
            if key & 0xFF == ord('q'):
                break

    video_loader.close()


def t_load_images(ds_root):

    path = os.path.join(ds_root, 'aun-lab', 'sth')

    for _, image, _, _ in LoadImages(path):
        if image is not None:
            image = np.transpose(image)
            cv2.imshow('Video Loader', image)
            key = cv2.waitKey(30)
            if key & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
        Image Loader: load single image, folder of images, images file list, video, and camera stream
    ''')
    parser.add_argument('path', help='path to image, images file, video, stream port')
    parser.add_argument('--load_mode', help='image, images_file, video, stream', default='stream')
    args = parser.parse_args()

    img_source = args.path
    # t_image_loader(ds_root)
    # t_load_images(ds_root)
    image_loader = ImageLoader(img_source, args.load_mode)
    if 'mav0' in img_source:
        image_loader = ImageLoaderEuRoC(img_source)

    while image_loader.is_ok():
        image = image_loader.get_next()

        if image is not None and image.frame is not None:
            cv2.imshow('Image', image.frame)
            k = cv2.waitKey(image_loader.fps) & 0xFF
            if k == ord('q') or k == 27:
                break
        else:
            break

    cv2.destroyAllWindows()

    image_loader.stop()
    image_loader.close()
