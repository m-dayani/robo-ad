"""
    Don't mix tflite with other dl packages to be able to run these independently
    You shouldn't use packages like: Tensorflow, Keras, Torch, ... in this file
    Install appropriate `numpy` version if encountered problems with tflite package
    Each model has a set of varied attributes which causes usage difficulties
    Both input and output details are different for various models (still don't know why???)
    todo: interpret the results of YOLO.tflite models differently (detection_dl)
    (there is a code sample for this but requires `torch`

    Average inference time on an Intel Core-i5 laptop for ssd-mobilenet is around 71 ms
    It's about 1269 ms on RPI 3B+ (Armv7l/Armhf)
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

sys.path.append('../../')
sys.path.append('../../data_loader')
sys.path.append('../../cv_mine/obj_det_tr')
from data_loader.image_loader import ImageLoader
from mviz.viz_ml import ML_Drawer
import tools.utils as fst
from cv_mine.obj_det_tr.obj_det import ObjDet, MyDetection, DetectionResults
from output.image_recorder import ImageRecorder
from tools.utils import MyTimer
import my_logging.my_logger as logger
my_logger = logger.setup_default_logger()


class ObjectDetectionTflite(ObjDet):
    def __init__(self, settings):
        super().__init__(settings)

        # settings_keys = settings.keys()

        if not os.path.isfile(self.model_path) or os.path.splitext(self.model_path)[-1] != '.tflite':
            print('ObjectDetectionTflite, Model is not accepted: ' + str(self.model_path))
            self.model_path = None

        self.tflite_interpreter = None
        if self.model_path is not None:
            self.tflite_interpreter = tflite.Interpreter(model_path=self.model_path)
            self.tflite_interpreter.allocate_tensors()

            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()

            self.height = self.input_details[0]['shape'][1]
            self.width = self.input_details[0]['shape'][2]
            self.input_size = (self.width, self.height)

            self.float_input = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # b, g, r
        self.gate_color = 2

        self.box_idx = 1
        self.score_idx = 0
        self.yolo_model = False
        if self.model_path is not None:
            if '1.tflite' in self.model_path:
                # for downloaded weights the indices are different
                self.box_idx = 0
                self.score_idx = 1
            if 'yolo' in self.model_path:
                self.yolo_model = True

        my_logger.info('ObjectDetectionTflite.__init__: model: ' + str(self.model_path) +
                       ', num labels: ' + str(len(self.labels)) + ', min_conf: ' + str(self.min_conf))

        print('ObjectDetectionTflite initialized successfully')

    def detect_ml(self, image):
        detections = []
        if self.tflite_interpreter is not None:
            new_img = cv2.resize(image, self.input_size)
            if self.float_input:
                new_img = (np.float32(new_img) - self.input_mean) / self.input_std

            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], [new_img])

            self.tflite_interpreter.invoke()
            rects = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
            scores = self.tflite_interpreter.get_tensor(self.output_details[2]['index'])

            for index, score in enumerate(scores[0]):
                if score > 0.5:
                    obj_name = self.labels[index]
                    detections.append([obj_name, score,
                                       self.calc_rect(rects[0][index], (image.shape[1], image.shape[0]))])
            return new_img

        return detections

    def resolve_output(self):
        interpreter = self.tflite_interpreter
        if self.yolo_model:
            res = interpreter.get_tensor(self.output_details[0]['index'])[0]
            boxes = res[:, :4]
            scores = np.squeeze(res[:, 4:5])
            classes = res[:, 5:]
        else:
            # Bounding box coordinates of detected objects
            boxes = interpreter.get_tensor(self.output_details[self.box_idx]['index'])[0]
            # Class index of detected objects
            classes = interpreter.get_tensor(self.output_details[3]['index'])[0]
            # Confidence of detected objects
            scores = interpreter.get_tensor(self.output_details[self.score_idx]['index'])[0]
        return boxes, scores, classes

    def detect(self, frame):

        detections = DetectionResults()
        if frame is None:
            return detections

        image = frame.frame
        ts = frame.ts

        interpreter = self.tflite_interpreter
        if interpreter is None:
            print('Warning: detection with None model')
            return detections

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(self.input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes, scores, classes = self.resolve_output()

        results = []
        if self.yolo_model:
            results = self.process_yolo_results(boxes, scores, classes, image.shape[:2])
        else:
            for i in range(len(scores)):
                score = scores[i]
                if self.min_conf < score <= 1.0:
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions,
                    # need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * img_h)))
                    xmin = int(max(1, (boxes[i][1] * img_w)))
                    ymax = int(min(img_h, (boxes[i][2] * img_h)))
                    xmax = int(min(img_w, (boxes[i][3] * img_w)))

                    cls_id = int(classes[i])
                    object_name = self.labels[cls_id]  # Look up object name from "labels" array using class index

                    det = MyDetection(cls_id=cls_id, name=object_name, bbox=[xmin, ymin, xmax, ymax], conf=score)
                    results.append(det)

        if len(results) > 0:
            detections = DetectionResults(ts=ts, results=results)

        return detections

    def process_yolo_results(self, boxes, scores, classes, img_size):
        results = []

        c_scores = scores > self.min_conf
        n_det = c_scores.sum()

        if n_det > 0:
            boxes = boxes[c_scores]
            scores = scores[c_scores]
            classes = classes[c_scores]

            boxes[:, 0] *= self.input_size[1]  # x
            boxes[:, 1] *= self.input_size[0]  # y
            boxes[:, 2] *= self.input_size[1]  # w
            boxes[:, 3] *= self.input_size[0]  # h

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            boxes = self.xywh2xyxy(boxes)

            # Apply non-max-suppression
            new_detections = self.non_max_suppression(boxes, scores, classes, 0.5)

            for det in new_detections:
                box = np.array(det[3:])
                box = np.int32(self.scale_coords(self.input_size, box, img_size).round())

                my_det = MyDetection(cls_id=det[0], name=det[1], bbox=box, conf=det[2])
                results.append(my_det)

        return results

    def get_bboxes(self, res):
        bboxes = dict()
        for r in res[0]:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                if self.tflite_interpreter.names[int(c)] not in bboxes.keys():
                    bboxes[self.tflite_interpreter.names[int(c)]] = []
                bboxes[self.tflite_interpreter.names[int(c)]].append(b.numpy())
        return bboxes

    @staticmethod
    def calc_rect(box, img_size):
        height, width = img_size
        y_min = int(max(1, (box[0] * height)))
        x_min = int(max(1, (box[1] * width)))
        y_max = int(min(height, (box[2] * height)))
        x_max = int(min(width, (box[3] * width)))

        return np.array([x_min, y_min, x_max, y_max])

    def non_max_suppression(self, boxes, scores, classes, nms_threshold):
        detections = []
        # Find unique classes:
        cls_idx_arr = np.argmax(classes, axis=1)
        unique_cls = np.unique(cls_idx_arr)
        for cls in unique_cls:
            c_cls_idx = cls_idx_arr == cls
            new_boxes = boxes[c_cls_idx]
            new_scores = scores[c_cls_idx]

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(bboxes=new_boxes, scores=new_scores,
                                       score_threshold=self.min_conf, nms_threshold=nms_threshold)

            # Filter out the boxes based on the NMS result
            # filtered_boxes = [new_boxes[i] for i in indices.flatten()]

            obj_name = self.labels[cls]

            for i in indices.flatten():
                box = new_boxes[i]
                detections.append([cls, obj_name, new_scores[i], box[0], box[1], box[2], box[3]])

        return detections

    @staticmethod
    def clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[0] = np.clip(boxes[0], 0, img_shape[1])  # x1
        boxes[1] = np.clip(boxes[1], 0, img_shape[0]) # y1
        boxes[2] = np.clip(boxes[2], 0, img_shape[1])  # x2
        boxes[3] = np.clip(boxes[3], 0, img_shape[0])  # y2

    @staticmethod
    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[[0, 2]] -= pad[0]  # x padding
        coords[[1, 3]] -= pad[1]  # y padding
        coords[:4] /= gain
        ObjectDetectionTflite.clip_coords(coords, img0_shape)
        return coords

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


def t_obj_detection(detector: ObjectDetectionTflite, image_loader: ImageLoader, save_dir):

    # detector.start_in_background()
    img_recorder = ImageRecorder(save_dir, fps=15)
    img_recorder.start_recording()

    area_img = image_loader.resolution[0] * image_loader.resolution[1]

    det_info = dict()

    my_timer = MyTimer(show_time=True)

    while image_loader.is_ok():

        ts_frame = image_loader.get_next()
        if ts_frame is None:
            break

        img_show = np.copy(ts_frame.frame)

        # reset the detectors frame
        # detector.set_last_frame(ts_frame)

        # get last results
        # results = detector.get_last_results()
        results = detector.detect(ts_frame)
        results = detector.refine_detections(results)

        if results is not None and results.results is not None:
            for det in results.results:
                center, area = det.calc_center_area()
                area_norm = float(area) / area_img
                label = det.name
                # print(label + ', ' + str(det.conf) + ', ' + str(center) + ', ' + str(area_norm))

                if label not in det_info.keys():
                    det_info[label] = dict()
                    det_info[label]['conf'] = []
                    det_info[label]['center'] = []
                    det_info[label]['area'] = []
                det_info[label]['conf'].append(det.conf)
                det_info[label]['center'].append(center)
                det_info[label]['area'].append(area_norm)

        # show the results
        ML_Drawer.draw_detections(img_show, results)

        # ts_frame.frame = img_show
        # img_recorder.set_last_frame(ts_frame)

        my_timer.roll(time.time_ns(), 'Average inference time for TFLite OD model: ')

        cv2.imshow('Detections', img_show)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    print('criteria: mean, std, min, max')
    for label in det_info.keys():
        print(label + ':')
        conf_arr = np.array(sorted(det_info[label]['conf']))
        print(f'confidence: {np.mean(conf_arr)}, {np.std(conf_arr)}, {np.min(conf_arr)}, {np.max(conf_arr)}')
        area_arr = np.array(sorted(det_info[label]['area']))
        print(f'area: {np.mean(area_arr)}, {np.std(area_arr)}, {np.min(area_arr)}, {np.max(area_arr)}')
        c_arr = np.array(det_info[label]['center'])
        print('center: ' + str(np.mean(c_arr, axis=0)) + ', ' + str(np.std(c_arr, axis=0)))

    img_recorder.stop_recording()
    detector.clean()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''
        Learning-based object detection (general Tensorflow and PyTorch models)
    ''')
    parser.add_argument('path_settings', help='settings file path')
    parser.add_argument('img_src', help='image source path (images, video, ...)')
    parser.add_argument('--img_load_mode', help='image load mode: video (default), image_folder, ...', default='video')
    parser.add_argument('--save_dir', help='output results path', default='')
    args = parser.parse_args()

    settings_file = args.path_settings
    settings = fst.load_settings(settings_file)

    img_src = args.img_src
    image_loader = ImageLoader(img_src, args.img_load_mode)

    save_dir = args.save_dir
    if len(save_dir) <= 0:
        save_dir = os.path.join(os.path.split(img_src)[0], 'detections')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    detector = ObjectDetectionTflite(settings)

    t_obj_detection(detector, image_loader, save_dir)

