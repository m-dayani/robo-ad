import os
import sys
import threading
import argparse
import time

import cv2
import numpy as np

sys.path.append('../')
sys.path.append('../data_loader/')
sys.path.append('../slam/vpr_dbow2/')
from cv_mine.obj_det_tr.lane_finder import LaneTracker
from cv_mine.obj_det_tr.detection_dl_tflite import ObjectDetectionTflite
from data_loader.tt_loader import TabularTextLoader
from data_loader.image_loader import ImageLoader
import tools.utils as fst
import imageproc
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


class AutoDriving:
    def __init__(self, settings):
        # self.img_loader = img_loader
        self.last_frame = None
        self.frame_lock = threading.Lock()

        self.ml_detector = ObjectDetectionTflite(settings)

        # lane detection modes: classic, cmd, vpr, ml_seg
        self.lane_dt_mode = settings['lane_dt_mode']

        self.lane_tracker = None
        self.cmd_player = None
        self.last_ts_state = None
        self.ts_delay_done = False
        self.cmd_table = None
        self.vpr_processor = None

        ds_root = settings['ds_root']

        # if self.lane_dt_mode == 'classic':
        ferrari_path = os.path.join(ds_root, 'images', 'ferrari.png')
        self.lane_tracker = LaneTracker(settings, ferrari_path)
        # print('AutoDriving loaded with classic lane detection')

        if self.lane_dt_mode == 'cmd':
            path_cmd = os.path.join(ds_root, settings['path_cmd'])
            self.cmd_player = TabularTextLoader(path_cmd)
            print('AutoDriving loaded with cmd: ' + str(path_cmd))

        elif self.lane_dt_mode == 'vpr':
            # racetrack
            self.path_cmd = os.path.join(ds_root, settings['path_id_cmd'])
            self.path_db = os.path.join(ds_root, settings['path_dbow_db'])

            # urban track
            self.path_cmd_urban = os.path.join(ds_root, settings['path_id_cmd_urban'])
            self.path_db_urban = os.path.join(ds_root, settings['path_dbow_db_urban'])

            self.load_vpr(self.path_cmd, self.path_db)

            print('AutoDriving loaded with VPR lane detection')

            # logging
            my_logger.info('AutoDriving.__init__: VPR, path_cmd: ' + str(self.path_cmd))
            my_logger.info('AutoDriving.__init__: VPR, path_db: ' + str(self.path_db))

        self.started = False
        self.started_lock = threading.Lock()

        self.thread_run = None
        self.default_state = np.zeros(4, dtype=np.uint8)
        self.default_state[:2] = 127
        self.state = np.zeros(8, dtype=np.uint8)
        self.state[:2] = np.copy(self.default_state[:2])

        # delay buffer for mismatch between lane detection and actual position in 'classic' mode
        self.sb_size = settings['state_buffer_size']
        self.state_buffer = np.repeat(self.default_state[:2].reshape(1, 2), self.sb_size, axis=0)

        # urban mode (traffic light) or race mode??
        self.ad_urban = False

        self.last_ts = -1
        self.last_cmd_ts = -1
        self.n_matches = settings['vpr_n_matches']

        # old (with center thresholding): [xmin, xmax, ymin, ymax, amin, amax]
        # new (only area): th_area
        self.od_lims = []
        path_od_lims = os.path.join(ds_root, settings['path_od_lims'])
        od_lims_data = TabularTextLoader(path_od_lims)
        for lims in od_lims_data.data:
            self.od_lims.append(np.float32(lims[0]))
        od_lims_data.close()

        self.stop_driving = False
        self.cnt_stop_check = 0
        self.max_stop_check = 15
        self.cnt_logger = 0

        print('AutoDriving initialized successfully')

    def load_vpr(self, path_cmd, path_db):
        cmd_data = TabularTextLoader(path_cmd)
        self.cmd_table = dict()
        for cmd in cmd_data.data:
            cmd_id = np.int32(cmd[0])
            ctrl = np.float32(cmd[1:])
            self.cmd_table[cmd_id] = ctrl
        # self.ts_cmd = np.array(sorted(self.cmd_table.keys()))
        cmd_data.close()

        self.vpr_processor = imageproc.ImageProcessor("", path_db)

    def set_urban(self, flag):
        self.ad_urban = flag
        # if flag:
        #     self.load_vpr(self.path_cmd_urban, self.path_db_urban)
        # else:
        #     self.load_vpr(self.path_cmd, self.path_db)

    def is_started(self):
        self.started_lock.acquire()
        flag = self.started
        self.started_lock.release()
        return flag

    def set_started(self, flag):
        self.started_lock.acquire()
        self.started = flag
        self.started_lock.release()

    def start_auto_driving(self):
        if not self.is_started():
            self.set_started(True)
            self.thread_run = threading.Thread(target=self.run)
            self.thread_run.start()
            self.ml_detector.start_in_background()

    def detect_lane(self, frame):

        if self.lane_dt_mode == 'classic':
            image = frame.frame

            # calculate the steering angle and trot
            _, steering_ang = self.lane_tracker.process(image, vis=False)
            # if confidence > 62.0:
            #     steer = steering_ang / 90 * 127.0 + 127.0
            #     if steer > 255:
            #         steer = 255
            #     if steer < 0:
            #         steer = 0
            #     trot = 127
            # else:
            #     steer = self.state[1]
            #     trot = 0

            # generate and send a command to Arduino
            if steering_ang < 0:
                self.state[:2] = self.default_state[:2]
            else:
                # convert steering angle (0~180) to num
                steering_num = steering_ang * 255.0 / 180.0
                # steering_strength = steering_num - 127
                # steering_num = steering_strength * 2 + 127
                self.state[0] = 255
                self.state[1] = np.uint8(steering_num)

            self.push_back_state()

        elif self.lane_dt_mode == 'cmd':
            if self.last_ts_state is None or self.ts_delay_done:
                ts_state = self.cmd_player.get_next()
            else:
                ts_state = self.last_ts_state

            if ts_state is None or len(ts_state) <= 0:
                self.stop()
            else:
                ts = time.time_ns()
                cmd_ts = np.ulonglong(ts_state[0])

                if self.last_cmd_ts >= 0:
                    last_cmd_ts_diff = cmd_ts - self.last_cmd_ts

                    if last_cmd_ts_diff <= ts - self.last_ts:
                        self.state[:2] = np.copy(ts_state[1:])

                        # print('CMD ts diff: %.2f ms' % (np.float64(last_cmd_ts_diff) * 1e-6))
                        # print('prog ts diff: %.2f ms' % (np.float64(ts - last_ts) * 1e-6))

                        last_cmd_ts = cmd_ts
                        last_ts = ts
                        self.ts_delay_done = True
                    else:
                        self.ts_delay_done = False

                    self.last_ts_state = ts_state
                else:
                    # init.
                    self.state[:2] = np.copy(ts_state[1:])
                    last_cmd_ts = cmd_ts
                    last_ts = ts
                    # self.last_ts_state = ts_state

        elif self.lane_dt_mode == 'vpr':

            image = frame.frame

            # image = frame
            # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_gray = self.lane_tracker.get_bird_eye_gray(image)
            # my_logger.info('AutoDriving.run: VPR, image shape: ' + str(img_gray.shape))
            results = self.vpr_processor.process_image(img_gray, self.n_matches)
            # print(results)
            # my_logger.info('AutoDriving.run: VPR, results: ' + str(results))

            avg_cmd = self.default_state[:2]

            if len(results) != 2 * self.n_matches:
                my_logger.info('AutoDriving.detect_lane, VPR results wrong length: ' + str(len(results)))
            else:
                ids = results[:self.n_matches]
                scores = np.float32(results[self.n_matches:]) / 100.0
                scores_rep = np.repeat(scores.reshape(-1, 1), 2, axis=1)
                scores_sum = np.sum(scores)
                avg_cmd = []
                for cmd_id in ids:
                    # check if we have consecutive frames
                    # find the image ts from ts-id map
                    if cmd_id in self.cmd_table.keys():
                        avg_cmd.append(self.cmd_table[cmd_id])
                        # print(self.state)
                        # if avg_cmd is None:
                        #     avg_cmd = np.copy(state)
                        #     cnt_cmd = 1
                        # else:
                        #     avg_cmd += np.copy(state)
                        #     cnt_cmd += 1

                if len(avg_cmd) > 0:
                    avg_cmd = np.array(avg_cmd)
                    # avg_cmd = np.mean(avg_cmd, axis=1)
                    # use weighted mean instead
                    avg_cmd = np.sum(avg_cmd * scores_rep, axis=0) / scores_sum
                else:
                    avg_cmd = self.default_state[:2]

            self.state[:2] = np.uint8(avg_cmd[:2])

    def push_back_state(self):
        self.state_buffer[:(self.sb_size - 1), :2] = np.copy(self.state_buffer[1:])
        self.state_buffer[-1, :2] = np.copy(self.state[:2])

    @staticmethod
    def check_stop_det(detection, lims, img_shape=(480, 640)):
        center, area = detection.calc_center_area()
        area /= (img_shape[0] * img_shape[1])
        if area > lims:
            return True
        return False

    def process_detections(self, det_dict, img_shape):

        # decide based in the center and area of the objects
        det_keys = det_dict.keys()
        # break_auto = False
        # img_shape = frame.frame.shape[:2]
        if self.ad_urban:
            # in the urban mode, stop sign and tl are important
            # stop if tl_red detected, and start again otherwise
            # tl_red: False -> tl not detected or tl_green/off/yellow
            objs = ['tl_red', 'ss_stop']

            # tl_red = False
            # if 'tl_red' in det_keys:
            #     det = det_dict['tl_red']
            #     tl_red = self.check_stop_det(det, self.od_lims[det.cls_id], img_shape=img_shape)
            # ss_stop = False
            # if 'ss_stop' in det_keys:
            #     det = det_dict['ss_stop']
            #     ss_stop = self.check_stop_det(det, self.od_lims[det.cls_id], img_shape=img_shape)
            #
            # if tl_red or ss_stop:
            #     self.stop_driving = True
            #     self.cnt_stop_check = 0
            # else:
            #     self.cnt_stop_check += 1
            #     if self.cnt_stop_check > self.max_stop_check:
            #         self.stop_driving = False
            #         self.cnt_stop_check = 0
        else:
            # in the race mode, human is important
            objs = ['bicycle', 'car', 'cow', 'horse', 'human', 'ss_parking', 'ss_stop']
            # if 'human' in det_keys:
            #     det = det_dict['human']
            #     self.stop_driving = self.check_stop_det(det, self.od_lims[det.cls_id], img_shape=img_shape)
            #     self.cnt_stop_check = 0
            # else:
            #     self.cnt_stop_check += 1
            #     if self.cnt_stop_check > self.max_stop_check:
            #         self.cnt_stop_check = 0
            #         self.stop_driving = False

            res_det = np.zeros(len(objs), dtype=np.bool_)
            for j, label in enumerate(objs):
                if label in det_keys:
                    mdet = det_dict[label]
                    if self.od_lims[mdet.cls_id] > 0:
                        res_det[j] = self.check_stop_det(mdet, self.od_lims[mdet.cls_id], img_shape=img_shape)

            if any(res_det):
                self.stop_driving = True
                self.cnt_stop_check = 0
            else:
                self.cnt_stop_check += 1
                if self.cnt_stop_check > self.max_stop_check:
                    self.stop_driving = False
                    self.cnt_stop_check = 0

            # the third state indicates which object is detected
            # if self.stop_driving:
            #     self.state[2] = 6
            # else:
            #     self.state[2] = 0

        # set detection bits
        det_bits = np.zeros(3, dtype=np.uint8)
        for k_det in det_keys:
            ddet = det_dict[k_det]
            cls_id = ddet.cls_id
            if 0 <= cls_id < 8:
                det_bits[2] |= (1 << cls_id)
            elif 8 <= cls_id < 16:
                det_bits[1] |= (1 << (cls_id - 8))
            elif 16 <= cls_id < 24:
                det_bits[0] |= (1 << (cls_id - 16))
        self.state[2:5] = np.copy(det_bits)

    def run(self):
        self.last_ts = -1
        self.last_cmd_ts = -1

        while True:
            # retrieve last frame:
            frame = self.get_last_frame()
            # self.state = np.copy(self.default_state)

            if frame is not None and frame.frame is not None:

                # detect objects
                self.ml_detector.set_last_frame(frame)
                detections = self.ml_detector.get_last_results()
                det_dict = self.ml_detector.refine_detections(detections, get_dict=True)

                # break if obstacle detected
                self.process_detections(det_dict, frame.frame.shape[:2])

                if self.stop_driving:
                    my_logger.info('AutoDriving.run: Break!')
                    self.state[0] = int(self.default_state[0])
                else:
                    # if self.ad_urban:
                    #     steering = 90.0
                    #     if self.lane_tracker is not None:
                    #         # get the steering angle from lane_tracker for better accuracy
                    #         _, steering = self.lane_tracker.process_contours2(frame.frame, vis=False)
                    #     steering_num = steering * 255 / 180.0
                    #     if self.cnt_logger % 100 == 0:
                    #         my_logger.info('AutoDriving.run: steering from lane_finder: ' +
                    #                        str(steering) + ', ' + str(steering_num))
                    #         self.cnt_logger = 0
                    #     self.cnt_logger += 1
                    #
                    #     self.state[0] = 255
                    #     self.state[1] = int(steering_num)
                    #     self.state[5] = 1
                    # else:
                    self.detect_lane(frame)
                    # if self.ad_urban:
                    #     self.state[0] = 255

            if not self.is_started():
                break

    def get_state(self):
        out_state = np.copy(self.state)
        if self.lane_dt_mode == 'classic':
            out_state[:2] = np.copy(self.state_buffer[0, :])
        return out_state

    def stop(self):
        if self.is_started():
            self.set_started(False)
            if self.thread_run is not None:
                self.thread_run.join(5.0)
                self.thread_run = None
            self.ml_detector.stop()

    def clean(self):
        if self.is_started():
            self.stop()
        if self.lane_dt_mode == 'cmd' and self.cmd_player is not None:
            self.cmd_player.close()
            self.cmd_player = None
        self.ml_detector.stop()
        self.ml_detector.clean()

    def set_last_frame(self, frame):
        self.frame_lock.acquire()
        self.last_frame = frame
        self.frame_lock.release()

    def get_last_frame(self):
        self.frame_lock.acquire()
        frame = self.last_frame
        self.frame_lock.release()
        return frame


def check_out_state(out_state, lim_speed, lim_angle):
    arduino_state = np.copy(out_state)

    # trottle
    if len(arduino_state) >= 0:
        speed = arduino_state[0]
        if speed < lim_speed[0]:
            arduino_state[0] = lim_speed[0]
        if speed > lim_speed[1]:
            arduino_state[0] = lim_speed[1]

    # steering
    if len(arduino_state) >= 1:
        angle = arduino_state[1]
        # convert angles from 0~255 to 60~120
        angle = angle / 255 * abs(lim_angle[1] - lim_angle[0]) + lim_angle[0]
        # print('Angle: ' + str(angle))
        if angle < lim_angle[0]:
            angle = lim_angle[0]
        if angle > lim_angle[1]:
            angle = lim_angle[1]
        arduino_state[1] = angle

    return arduino_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
            Autonomous Driving Visual Perception
        ''')
    parser.add_argument('path_settings', help='settings file path')
    parser.add_argument('img_src', help='images source')
    parser.add_argument('--load_mode', help='image loader load mode (default: video)', default='video')
    args = parser.parse_args()

    settings_file = args.path_settings
    settings = fst.load_settings(settings_file)

    path_video = args.img_src
    image_loader = ImageLoader(path_video, args.load_mode)

    auto_driving = AutoDriving(settings)
    if 'urban' in path_video:
        auto_driving.ad_urban = True

    auto_driving.start_auto_driving()

    lim_speed = [53, 203]
    lim_angle = [60, 120]

    draw_offset = 50

    while image_loader.is_ok():
        frame = image_loader.get_next()
        if frame is None:
            break
        auto_driving.set_last_frame(frame)

        if frame.frame is not None:
            img_show = np.copy(frame.frame)

            state = auto_driving.get_state()
            state_ard = check_out_state(state, lim_speed, lim_angle)

            detections = []
            for i in range(8):
                det_i = (state[4] >> i) & 0x01
                if det_i:
                    detections.append(auto_driving.ml_detector.labels[i])
            for i in range(8):
                det_i = (state[3] >> i) & 0x01
                if det_i:
                    detections.append(auto_driving.ml_detector.labels[i + 8])
            for i in range(3):
                det_i = (state[2] >> i) & 0x01
                if det_i:
                    detections.append(auto_driving.ml_detector.labels[i + 16])

            y1 = draw_offset + int(40.0 * state[0] / 255.0)
            y2 = draw_offset
            x_off = 320
            steer_ang = state_ard[1]
            m = np.tan(steer_ang * np.pi / 180.0)
            x1 = x_off + y1 / m
            x2 = x_off + y2 / m
            pt1 = np.int32([x1, y1])
            pt2 = np.int32([x2, y2])

            for i, det in enumerate(detections):
                cv2.putText(img_show, det, (540, draw_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if auto_driving.stop_driving:
                msg = 'Break (' + str(auto_driving.state[0]) + ')!'
                img_show = cv2.putText(img_show, msg, (40, draw_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.arrowedLine(img_show, pt1, pt2, (255, 0, 0), 3)

            cv2.imshow('AD Image', img_show)
            k = cv2.waitKey(24) & 0xFF

            if k == ord('q'):
                break

    auto_driving.stop()
    auto_driving.clean()
