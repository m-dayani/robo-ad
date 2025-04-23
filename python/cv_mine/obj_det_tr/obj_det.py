"""
    Base class for all object detection methods
"""
import copy
import threading
import os
import time
import yaml
import sys

import numpy as np

sys.path.append('../')
from tools.utils import MyTimer
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


class MyDetection(object):
    def __init__(self, cls_id=0, name='', bbox=None, conf=0.0):
        # self.ts = -1
        self.name = name  # label
        self.bbox = bbox
        self.conf = conf
        self.cls_id = cls_id

    def calc_center_area(self):
        bbox = np.array(self.bbox)
        center = 0.5 * (bbox[:2] + bbox[2:4])
        w, h = bbox[2:4] - bbox[:2]
        area = w * h
        return center, area


class DetectionResults(object):
    def __init__(self, ts=-1, results=None):
        self.results = results
        self.ts = ts


class ObjDet(object):
    def __init__(self, settings):
        # settings contains important info about model and ...
        settings_keys = settings.keys()

        # model path
        ds_root = settings['ds_root']
        self.model_path = os.path.join(ds_root, settings['model_path'])
        if not os.path.isfile(self.model_path):
            self.model_path = None

        # resolve labels
        self.labels = None
        if 'labels' in settings_keys:
            self.labels = settings['labels']
        elif 'labels_path' in settings_keys:
            # Load the labels map into memory
            labels_path = os.path.join(ds_root, settings['labels_path'])
            yaml_file = os.path.splitext(labels_path)[-1] == '.yaml'
            if yaml_file:
                with open(labels_path, 'r') as f:
                    # class names (assume COCO)
                    self.labels = yaml.load(f, Loader=yaml.FullLoader)['names']
            else:
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]

        self.min_conf = 0.5
        if 'min_conf' in settings_keys:
            self.min_conf = settings['min_conf']

        self.bg_thread = None

        self.accept_frame_lock = threading.Lock()
        self.last_frame = None

        self.started = False
        self.run_lock = threading.Lock()

        self.results = None
        self.results_lock = threading.Lock()

        self.timer = MyTimer()

    def is_running(self):
        self.run_lock.acquire()
        flag = self.started
        self.run_lock.release()
        return flag

    def set_started(self, flag):
        self.run_lock.acquire()
        self.started = flag
        self.run_lock.release()

    def start_in_background(self):
        if self.bg_thread is not None:
            self.stop()

        self.set_started(True)
        self.bg_thread = threading.Thread(target=self.play)
        self.bg_thread.start()

    def stop(self):
        if self.bg_thread is not None:
            self.set_started(False)
            self.bg_thread.join(5)
            self.bg_thread = None

    def set_last_frame(self, ts_frame):
        self.accept_frame_lock.acquire()
        self.last_frame = copy.deepcopy(ts_frame)
        self.accept_frame_lock.release()

    def get_last_frame(self):
        self.accept_frame_lock.acquire()
        frame = copy.deepcopy(self.last_frame)
        self.accept_frame_lock.release()
        return frame

    def set_last_results(self, results):
        self.results_lock.acquire()
        self.results = copy.deepcopy(results)
        self.results_lock.release()

    def get_last_results(self):
        self.results_lock.acquire()
        results = copy.deepcopy(self.results)
        self.results_lock.release()
        return results

    def play(self):

        while True:
            frame = self.get_last_frame()
            if frame is not None:
                results = self.detect(frame)
                self.set_last_results(results)
                self.timer.roll(time.time_ns())
                if self.timer.t_cnt % self.timer.show_rate == 0:
                    my_logger.info('obj_det.ObjDet.play: Average detection time: ' + str(self.timer.t_avg * 1e-6) + ' (ms)')
                    n_results = 0
                    if results is not None and results.results is not None:
                        n_results = len(results.results)
                    my_logger.info('obj_det.ObjDet.play: Num. results: ' + str(n_results))
            if not self.is_running():
                break

    def detect(self, frame):
        return DetectionResults()

    # def show(self, frame, detections: DetectionResults):
    #     for det in detections.results:
    #         bbox = np.int32(det.bbox)

    def clean(self):
        if self.bg_thread is not None:
            self.stop()

    @staticmethod
    def refine_detections(results: DetectionResults, get_dict=False):
        if results is None or results.results is None or len(results.results) <= 0:
            if get_dict:
                return dict()
            return results

        res_refined = DetectionResults(ts=results.ts, results=[])

        det_dict = dict()
        for det in results.results:
            label = det.name

            if label not in det_dict.keys():
                det_dict[label] = det
            else:
                pre_det = det_dict[label]
                if pre_det.conf < det.conf:
                    det_dict[label] = det

        if get_dict:
            return det_dict

        for det_key in det_dict.keys():
            res_refined.results.append(det_dict[det_key])

        return res_refined