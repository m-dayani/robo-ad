# Detect: Checkpoint lines, April tags, and Road/Traffic Signs
# Input: processed/unprocessed image
# Output: one or more (list) object instances containing some attributes: label, box, center, [distance to body]

"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""

# First version of square detector
# 1. calculate corners
# 2. optionally calculate the ROI by template matching
# 3. sort by response
# 4. perform a kind of non-max suppression:
#   a) pick the best response
#   b) remove all other points with a dist less than a th
#   c) group items in b) under the best response to average
#   d) when all points are classified, look for the structure: which points make a rectangle
#   e) can optionally refine the results by ROI

import glob
import math
import os
import sys
import time

import apriltag
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

sys.path.append('../')
sys.path.append('../../')
from cv_mine.cv_utils import img_resize, point_dist, get_mesh, smooth, calc_object_center
import tools.utils as fst   # my_tools
from cv_mine.cv_utils import color_thresh
from obj_det import ObjDet  # , MyDetection, DetectionResults


glob_frame = None


def mouse_callback(event, x, y, flags, params):
    global glob_frame
    if event == 1:
        if params == 0:
            if glob_frame is not None:
                print(f'{x}, {y}')
                print(glob_frame[y, x])


class ObjectDetection(ObjDet):
    def __init__(self, settings):
        super().__init__(settings)

        self.apriltag_family = settings['apriltag_family']
        options = apriltag.DetectorOptions(families=self.apriltag_family)
        self.april_detector = apriltag.Detector(options)
        self.april_scale = float(settings['apriltag_scale'])

        # b, g, r
        self.gate_color = 2

    @staticmethod
    def detect_line(image):
        dst = cv2.Canny(image, 50, 200, None, 3)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        return cdst

    @staticmethod
    def detect_line_p(image, vis=False):
        dst = cv2.Canny(image, 50, 200, None, 3)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        cdstP = image

        if not vis:
            return linesP, cdstP

        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

        return linesP, cdstP

    def detect_lines(self, image, filtering='median', edge='canny'):

        wobjs_list = []

        if len(image.shape) > 2:
            # perform line detection for each channel separately
            B, G, R = cv2.split(image)
            blue_lines = self.detect_lines(B)
            green_lines = self.detect_lines(G)
            red_lines = self.detect_lines(R)

            for bl in blue_lines:
                bl.color = 'blue'
                wobjs_list.append(bl)
            for gl in green_lines:
                gl.color = 'green'
                wobjs_list.append(gl)
            for rl in red_lines:
                rl.color = 'red'
                wobjs_list.append(rl)

        else:
            # filtering
            img_smooth = image
            if filtering == 'median':
                img_smooth = cv2.medianBlur(image, 3)
            elif filtering == 'gaussian':
                img_smooth = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)

            # edge detection
            # if edge == 'thresh':
            #     # todo detect horizontal/vertical/slanted lines
            #     img_edge = cvu.get_combined_gradients(img_smooth, (35, 100), (30, 255), (30, 255), (0.7, 1.3))
            # else:
            img_edge = cv2.Canny(img_smooth, 50, 200, None, 3)

            # detect lines
            lines_p = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 50, None, 50, 10)

            # if lines_p is not None:
            #     for line in lines_p:
            #         line_obj = world_objects.Line()
            #         line_obj.set_object(line.squeeze())
            #         wobjs_list.append(line_obj)

        return wobjs_list

    def detect_apriltag(self, image, vis=False):
        # this is very slow!
        gray = image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = self.april_detector.detect(gray)
        # print('detected ' + str(len(results)) + ' Apriltags in image')

        if not vis:
            return results, image

        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8") + ' #' + str(r.tag_id)
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # print("[INFO] tag family: {}".format(tagFamily))

        return results, image

    def detect_gate_canny(self, img, vis=False):

        is_color_img = len(img.shape) == 3

        # minimal smoothing
        # img_med = cv2.medianBlur(img, 5)
        # img_gauss = cv2.GaussianBlur(img_med, (3, 3), cv2.BORDER_DEFAULT)
        # color spaces
        img_gray = img
        if is_color_img:
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.split(img)[self.gate_color]

        # edge detection
        img_canny = cv2.Canny(img_gray, 127, 255)
        # morphology
        img_dilated = cv2.dilate(img_canny, (3, 3), iterations=5)
        img_eroded = cv2.erode(img_dilated, (3, 3), iterations=5)
        # detect dominant contours
        contours, h = cv2.findContours(img_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        img_blank = np.zeros(img.shape, dtype=np.uint8)
        if not is_color_img:
            img_blank = cv2.merge((img_blank, img_blank, img_blank))

        colors = np.random.randint(100, 255, size=(len(contours), 3), dtype=np.uint8)
        for contour, color in zip(contours, colors):
            contour_area = cv2.contourArea(contour)
            if contour_area > 255:
                color = (0, 0, 255)
            elif contour_area > 127:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            # cv2.drawContours(img_blank, [contour], -1, (int(color[0]), int(color[1]), int(color[2])), 2)
        # img_blank = cv2.drawContours(img_blank, contours, -1, (0, 0, 255), 2)
        # print(len(contours))
        # edge-based mean-shift

        # if len(contours_sorted) > 0:
        #     cnt = contours_sorted[0]
        #     peri = cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        #     if len(approx) == 4:
        #         return cv2.drawContours(img_blank, [approx], -1, (255, 0, 0), 2)

        hLines, img_show = self.detect_line_p(img, vis)

        return img_show

    @staticmethod
    def get_non_max_array(arr, th_dist=30):
        max_idx = np.argmax(arr)
        # max_val = arr[max_idx]
        arr[max_idx] = 0
        max_idx1 = np.argmax(arr)
        avg_max = max_idx
        cnt = 1
        while abs(max_idx1 - max_idx) < th_dist:
            arr[max_idx1] = 0
            max_idx1 = np.argmax(arr)
            avg_max += max_idx1
            cnt += 1
        max_val1 = arr[max_idx1]
        avg_max /= cnt
        print("(max_idx, avg_max) = ({}, {})".format(max_idx, avg_max))
        return max_idx, avg_max, arr

    def detect_gate_lines(self, img, vis=False):

        # minimal smoothing
        img_med = cv2.medianBlur(img, 5)
        img_gauss = cv2.GaussianBlur(img_med, (3, 3), cv2.BORDER_DEFAULT)

        # edge detection
        img_th = color_thresh(img_gauss, (30, 100, 20, 70, 80, 160))

        # morphology
        kernel_size = (11, 11)
        img_dilated = cv2.dilate(img_th, kernel_size, iterations=5)
        img_eroded = cv2.erode(img_dilated, kernel_size, iterations=5)

        lines_p, img_show = self.detect_line_p(img_eroded, vis=True)

        if lines_p is not None:

            th_angle = 45
            th_dist = 20

            list_angles = np.array([-90, -45, 0, 45, 90])
            results = {-90: [], -45: [], 0: [], 45: []}

            for line in lines_p:
                dline = line[0, 2:] - line[0, :2]
                theta = np.arctan2(dline[1], dline[0]) / np.pi * 180
                idx_angle = np.argmin(abs(list_angles - theta))
                if idx_angle == 4:
                    idx_angle = 0
                print(theta)

                results[list_angles[idx_angle]].append(line.squeeze())

            # merge similar lines:
            merged_lines = list()
            for key in results.keys():
                list_lines = results[key]
                # for line in list_lines:
                #     if key == -90:

            # # find the first two vertical maximum
            # hist_x = np.sum(img_eroded, axis=0)
            # max_x_idx0, avg_max_x0, new_hist_x = ObjectDetection.get_non_max_array(hist_x)
            # max_x_idx1, avg_max_x1, _ = ObjectDetection.get_non_max_array(new_hist_x)
            #
            # # do the same for horizontal direction
            # hist_y = np.sum(img_eroded, axis=1)
            # max_y_idx0, avg_max_y0, new_hist_y = ObjectDetection.get_non_max_array(hist_y)
            # max_y_idx1, avg_max_y1, _ = ObjectDetection.get_non_max_array(new_hist_y)
            #
            # h, w = img_eroded.shape[:2]
            #
            # img_show = cv2.line(img_show, np.int32((avg_max_x0, 0)), np.int32((avg_max_x0, h)), (255, 0, 0), 2)
            # img_show = cv2.line(img_show, np.int32((avg_max_x1, 0)), np.int32((avg_max_x1, h)), (255, 0, 0), 2)
            # img_show = cv2.line(img_show, np.int32((0, avg_max_y0)), np.int32((w, avg_max_y0)), (255, 0, 0), 2)
            # img_show = cv2.line(img_show, np.int32((0, avg_max_y1)), np.int32((w, avg_max_y1)), (255, 0, 0), 2)

            # bins_x = np.linspace(0, len(hist_x) - 1, len(hist_x))
            # bins_y = np.linspace(0, len(hist_y) - 1, len(hist_y))
            #
            # plt.figure(figsize=(15, 15))
            # plt.subplot(1, 2, 1)
            # plt.plot(bins_x, hist_x)
            # plt.subplot(1, 2, 2)
            # plt.plot(bins_y, hist_y)
            # plt.show()

        return img_show

    @staticmethod
    def detect_gate_contours(img, vis=False):

        # minimal smoothing
        img_med = cv2.medianBlur(img, 5)
        img_gauss = cv2.GaussianBlur(img_med, (3, 3), cv2.BORDER_DEFAULT)

        # edge detection
        b, g, r = cv2.split(img_gauss)
        x1 = np.bitwise_and(30 < b, b < 100)
        x2 = np.bitwise_and(20 < g, g < 70)
        x3 = np.bitwise_and(80 < r, r < 160)
        bg = np.bitwise_and(x1, x2)
        img_th = np.uint8(np.bitwise_and(bg, x3)) * 255
        # morphology
        kernel_size = (11, 11)
        img_dilated = cv2.dilate(img_th, kernel_size, iterations=5)
        img_eroded = cv2.erode(img_dilated, kernel_size, iterations=5)

        # img_show, contours, hierarchy = cv2.findContours(img_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, h = cv2.findContours(img_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blank = np.zeros(img.shape)
        blank = cv2.drawContours(blank, contours, -1, (0, 0, 255), 2)

        return blank

    @staticmethod
    def detect_gate_gravity(img, vis=False):

        # minimal smoothing
        img_med = cv2.medianBlur(img, 5)
        img_gauss = cv2.GaussianBlur(img_med, (3, 3), cv2.BORDER_DEFAULT)

        # edge detection
        b, g, r = cv2.split(img_gauss)
        x1 = np.bitwise_and(30 < b, b < 100)
        x2 = np.bitwise_and(20 < g, g < 70)
        x3 = np.bitwise_and(80 < r, r < 160)
        bg = np.bitwise_and(x1, x2)
        img_th = np.uint8(np.bitwise_and(bg, x3)) * 255
        # morphology
        kernel_size = (11, 11)
        img_dilated = cv2.dilate(img_th, kernel_size, iterations=5)
        img_eroded = cv2.erode(img_dilated, kernel_size, iterations=5)

        Y, X = np.where(img_eroded >= 1)

        center = (np.sum(X) / len(X), np.sum(Y) / len(Y))

        return center

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

    def detect_gate_ml(self, img, vis=False, min_conf=0.5):
        bboxes = []
        detections = []
        imH, imW, _ = img.shape
        if self.tflite_interpreter is not None:
            # new_img = cv2.resize(img, self.ml_size)
            new_img = cv2.resize(img, (self.width, self.height))
            input_data = np.expand_dims(new_img, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.float_input:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # if self.model_type == 'yolo_v5' or self.model_type == 'mobilenet':
            #     self.tflite_interpreter.set_tensor(self.input_details[0]['index'], [np.float32(new_img)])
            # else:
            #     self.tflite_interpreter.set_tensor(self.input_details[0]['index'], [new_img])

            self.tflite_interpreter.invoke()

            # Retrieve detection results
            boxes = self.tflite_interpreter.get_tensor(self.output_details[1]['index'])[
                0]  # Bounding box coordinates of detected objects
            classes = self.tflite_interpreter.get_tensor(self.output_details[3]['index'])[0]  # Class index of detected objects
            scores = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])[0]  # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = 'gate' #labels[int(classes[i])]  # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(img, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(img, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

                    detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
                    bboxes.append(np.array([xmin, ymin, xmax, ymax]))

            # All the results have been drawn on the image, now display the image
            # if txt_only == False:  # "text_only" controls whether we want to display the image results or just save them in .txt files
            #     image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     plt.figure(figsize=(12, 16))
            #     plt.imshow(image)
            #     plt.show()

            # Save detection results in .txt files (for calculating mAP)
            # elif txt_only == True:
            #
            #     # Get filenames and paths
            #     image_fn = os.path.basename(image_path)
            #     base_fn, ext = os.path.splitext(image_fn)
            #     txt_result_fn = base_fn + '.txt'
            #     txt_savepath = os.path.join(savepath, txt_result_fn)
            #
            #     # Write results to text file
            #     # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            #     with open(txt_savepath, 'w') as f:
            #         for detection in detections:
            #             f.write('%s %.4f %d %d %d %d\n' % (
            #                 detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

            # rects = self.tflite_interpreter.get_tensor(
            #     self.output_details[0]['index'])
            # scores = self.tflite_interpreter.get_tensor(
            #     self.output_details[2]['index'])
            #
            # # print("For file {}".format(file.stem))
            # # print("Rectangles are: {}".format(rects))
            # # print("Scores are: {}".format(scores))
            # for index, score in enumerate(scores):
            #     if score > 0.5:
            #         box = rects[index]
            #         if vis:
            #             new_img = self.draw_rect(new_img, box)
            #         bboxes.append(self.calc_rect(box, img.shape[:2]))

            return bboxes, img, detections

        return bboxes, img, detections

    @staticmethod
    def intersections(edged):
        # Height and width of a photo with a contour obtained by Canny
        h, w = edged.shape

        hl = cv2.HoughLines(edged, 2, np.pi / 180, 190)[0]
        # Number of lines. If n!=4, the parameters should be tuned
        n = hl.shape[0]

        # Matrix with the values of cos(theta) and sin(theta) for each line
        T = np.zeros((n, 2), dtype=np.float32)
        # Vector with values of rho
        R = np.zeros((n), dtype=np.float32)

        T[:, 0] = np.cos(hl[:, 1])
        T[:, 1] = np.sin(hl[:, 1])
        R = hl[:, 0]

        # Number of combinations of all lines
        c = n * (n - 1) / 2
        # Matrix with the obtained intersections (x, y)
        XY = np.zeros((c, 2))
        # Finding intersections between all lines
        for i in range(n):
            for j in range(i + 1, n):
                XY[i + j - 1, :] = np.linalg.inv(T[[i, j], :]).dot(R[[i, j]])

        # filtering out the coordinates outside the photo
        XY = XY[(XY[:, 0] > 0) & (XY[:, 0] <= w) & (XY[:, 1] > 0) & (XY[:, 1] <= h)]
        # XY = order_points(XY) # obtained points should be sorted
        return XY

    @staticmethod
    def calc_rect(box, img_size):
        height, width = img_size
        y_min = int(max(1, (box[0] * height)))
        x_min = int(max(1, (box[1] * width)))
        y_max = int(min(height, (box[2] * height)))
        x_max = int(min(width, (box[3] * width)))

        return np.array([x_min, y_min, x_max, y_max])

    @staticmethod
    def draw_rect(image, box):
        bbox = ObjectDetection.calc_rect(box, image.shape[:2])

        # draw a rectangle on the image
        return cv2.rectangle(image, bbox[:2], bbox[2:], (255, 0, 255), 2)


def main_lines(argv):
    default_file = 'data/image0.png'    # 'data/sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv2.waitKey()
    return 0


def my_detect_lines(src):
    if src is None:
        print('Input image is not provided')
        return -1

    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    return cdstP


def main_circle(argv):
    default_file = 'data/image_ball.jpg'    # 'data/smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    scale = 0.5
    src = cv2.resize(src, (int(scale * src.shape[1]), int(scale * src.shape[0])),  cv2.INTER_AREA)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # gray = cv.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 5), 2, 2)

    dst = cv2.Canny(gray, 50, 200, None, 3)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 18,
                              param1=60, param2=27,
                              minRadius=30, maxRadius=70)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)

    cv2.imshow("detected circles", src)
    cv2.waitKey(0)

    return 0


def find_circles(img, hparams=(18, 60, 27, 30, 70)):

    gray = img
    if len(gray.shape) >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / hparams[0],
                              param1=hparams[1], param2=hparams[2],
                              minRadius=hparams[3], maxRadius=hparams[4])
    return circles


def main_hough(argv):
    default_file = 'assets/sudoku.png'
    default_file1 = 'assets/tag_and_board.jpg'
    filename = argv[0] if len(argv) > 0 else default_file1
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv2.waitKey()
    return 0


def main_april():
    settings_file = 'config/settings.yaml'
    settings = fst.load_settings(settings_file)

    lane_tracker = ObjectDetection(settings)

    image_file = 'assets/apriltagrobots_overlay.webp'
    image_file1 = 'assets/182070099583581.png'
    image_file2 = 'assets/tag_and_board.jpg'
    image = cv2.imread(image_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8") + ' #' + str(r.tag_id)
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def t_basic_use(detector, image):
    result = detector.detect_line(image)
    cv2.imshow("OpenCV Line", result)
    cv2.waitKey()

    result = detector.detect_line_p(image)
    cv2.imshow("OpenCV LineP", result)
    cv2.waitKey()

    wobjs, result = detector.detect_apriltag(image, True)
    cv2.imshow("Apriltag", result)
    cv2.waitKey()

    result = detector.detect_ml(image)
    cv2.imshow("TFLite Object Detection", result)
    cv2.waitKey()


def t_ml_video(detector, video_file):
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():

        ret, image = cap.read()

        if not ret:
            break

        result = detector.detect_ml(image)
        cv2.imshow("TFLite Object Detection", result)
        cv2.waitKey(1)

    cap.release()


def t_detect_gates_ml(detector, video_file):
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():

        ret, image = cap.read()

        if not ret:
            break

        gates, result, _ = detector.detect_gate_ml(image, vis=True)
        cv2.imshow("TFLite Object Detection", result)
        cv2.waitKey(1)

    cap.release()


def t_april_stream(settings, detector, image_folder):
    cam0_images = glob.glob(image_folder + '/cam0/data/*')
    cam0_images = sorted(cam0_images)

    intrinsics0 = settings['intrinsics0']
    K0 = np.array([[intrinsics0[0], 0.0, intrinsics0[2]],
                   [0.0, intrinsics0[1], intrinsics0[3]],
                   [0.0, 0.0, 1.0]])
    D0 = np.array(settings['dist_coeffs0'])

    avg_imread = 0.0
    avg_undist = 0.0
    avg_april = 0.0
    cnt = 0
    for cam0_img in cam0_images:
        t0 = time.time_ns()
        img0 = cv2.imread(cam0_img, cv2.IMREAD_UNCHANGED)

        t1 = time.time_ns()
        img0_ud = cv2.undistort(img0, K0, D0, None)

        t2 = time.time_ns()
        wobjs, img2 = detector.detect_apriltag(img0_ud)

        t3 = time.time_ns()
        avg_imread += (t1 - t0) * 1e-6
        avg_undist += (t2 - t1) * 1e-6
        avg_april += (t3 - t2) * 1e-6
        cnt += 1

    print("Average imread (ms): %.3f" % (avg_imread / cnt))
    print("Average undistort (ms): %.3f" % (avg_undist / cnt))
    print("Average april detection (ms): %.3f" % (avg_april / cnt))


def t_april_video(detector, video_file):
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        wobjs, img2 = detector.detect_apriltag(frame, vis=True)

        cv2.imshow("Apriltag", img2)
        cv2.waitKey(30)

    cap.release()

    # print("Average imread (ms): %.3f" % (avg_imread / cnt))
    # print("Average undistort (ms): %.3f" % (avg_undist / cnt))
    # print("Average april detection (ms): %.3f" % (avg_april / cnt))


def t_april_video1(settings, detector, video_path):

    cam = cv2.VideoCapture(video_path)

    intrinsics0 = settings['intrinsics0']
    K0 = np.array([[intrinsics0[0], 0.0, intrinsics0[2]],
                        [0.0, intrinsics0[1], intrinsics0[3]],
                        [0.0, 0.0, 1.0]])
    D0 = np.array(settings['dist_coeffs0'])

    avg_imread = 0.0
    avg_undist = 0.0
    avg_april = 0.0
    cnt = 0
    while cam.isOpened():

        t0 = time.time_ns()
        ret, img0 = cam.read()
        img0 = img0[:, :2]
        if not ret:
            break

        t1 = time.time_ns()
        img0_ud = cv2.undistort(img0, K0, D0, None)

        t2 = time.time_ns()
        wobjs, img2 = detector.detect_apriltag(img0_ud)

        t3 = time.time_ns()
        avg_imread += (t1 - t0) * 1e-6
        avg_undist += (t2 - t1) * 1e-6
        avg_april += (t3 - t2) * 1e-6
        cnt += 1

        cv2.imshow("Apriltag", img2)
        cv2.waitKey(30)

    cam.release()


def t_lines(settings, detector, image_folder):
    cam0_images = glob.glob(image_folder + '/cam0/data/*')
    cam0_images = sorted(cam0_images)

    intrinsics0 = settings['intrinsics0']
    K0 = np.array([[intrinsics0[0], 0.0, intrinsics0[2]],
                   [0.0, intrinsics0[1], intrinsics0[3]],
                   [0.0, 0.0, 1.0]])
    D0 = np.array(settings['dist_coeffs0'])

    for cam0_img in cam0_images:

        # t0 = time.time_ns()
        img0 = cv2.imread(cam0_img, cv2.IMREAD_UNCHANGED)
        img0 = cv2.undistort(img0, K0, D0)

        lines = detector.detect_lines(img0, edge='thresh')

        img_show = img0
        if len(img0.shape) < 3:
            img_show = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

        for line in lines:
            color = (255, 0, 255)
            if line.color == 'blue':
                color = (255, 0, 0)
            elif line.color == 'green':
                color = (0, 255, 0)
            elif line.color == 'red':
                color = (0, 0, 255)

            l = line.line
            cv2.line(img_show, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)

        cv2.imshow('Lines', img_show)
        cv2.waitKey(1)


def t_gate_detector(detector, video_file):
    cap = cv2.VideoCapture(video_file)

    cnt = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        if cnt < 200:
            cnt += 1
            continue

        img2 = detector.detect_gate_contours(frame, vis=True)
        # img2 = detector.detect_gate_lines(frame, vis=True)
        print(detector.detect_gate_gravity(frame, vis=True))

        cv2.imshow("Gates", img2)
        cv2.waitKey(30)

    cap.release()


def t_color_detection(video_file):
    global glob_frame

    window_width, window_height = (640, 480)
    cv2.namedWindow('image0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image0', window_width, window_height)
    cv2.setMouseCallback('image0', mouse_callback, 0)

    cap = cv2.VideoCapture(video_file)

    cnt = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        glob_frame = np.copy(frame)

        cv2.imshow("image0", frame)
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


def detect_black_square(image, patch):
    # Template Matching (find ROI)
    tm_res = cv2.matchTemplate(image, patch, cv2.TM_SQDIFF_NORMED)
    _, _, tm_loc, _ = cv2.minMaxLoc(tm_res)

    pt1 = np.array([tm_loc[0], tm_loc[1]])
    pt2 = pt1 + np.array([patch.shape[1], patch.shape[0]])

    # mask = np.zeros(image.shape)
    # mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 1.0

    # Detect Harris Corners
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    scores = dict()
    for r in range(pt1[1], pt2[1]):
        for c in range(pt1[0], pt2[0]):
            scores[dst[r, c]] = (r, c)

    l = list(scores.items())
    l.sort(reverse=True)  # sort in reverse order
    scores = dict(l)

    th_res = 0.1
    th_px = 2
    best_res = -1
    idx = 0
    res = dict()
    for score in scores.keys():

        if best_res < score:
            best_res = score
        elif score < th_res * best_res:
            break

        loc = np.array(scores[score])

        matched_idx = -1
        for pt in res.keys():
            pt_loc = res[pt]['avg']
            d = point_dist(loc, pt_loc)

            if d < th_px:
                matched_idx = pt
                break

        if matched_idx >= 0:
            # append point to list
            curr_n = res[matched_idx]['n']
            res[matched_idx]['loc'].append(loc)
            res[matched_idx]['avg'] = (res[matched_idx]['avg'] * curr_n + loc) / (curr_n + 1)
            res[matched_idx]['n'] = len(res[matched_idx]['loc'])
        else:
            # create new group
            res[idx] = dict()
            res[idx]['loc'] = []
            res[idx]['loc'].append(loc)
            res[idx]['n'] = len(res[idx]['loc'])
            res[idx]['avg'] = loc
            idx += 1

    points = []
    for key in res.keys():
        points.append(res[key]['avg'])
        if len(points) >= 4:
            break
    points = np.array(points)

    # Structural Analysis
    if len(points) < 4:
        return []

    # which point is the origin, x-axis, and y-axis
    # determine the x-axis
    x_idx = 0
    dx_arr = [abs(points[:, 1] - 0), abs(points[:, 1] - image.shape[1])]
    if min(dx_arr[1]) < min(dx_arr[0]):
        x_idx = 1
    y_idx = 0
    dy_arr = [abs(points[:, 0] - 0), abs(points[:, 0] - image.shape[0])]
    if min(dy_arr[1]) < min(dy_arr[0]):
        y_idx = 1

    orig_idx = np.argmin(dx_arr[x_idx] * dy_arr[y_idx])

    dx_orig = abs(points[:, 1] - points[orig_idx, 1])
    dx_orig[orig_idx] += max(dx_orig)
    y_idx = np.argmin(dx_orig)

    dy_orig = abs(points[:, 0] - points[orig_idx, 0])
    dy_orig[orig_idx] += max(dy_orig)
    x_idx = np.argmin(dy_orig)

    xy_idx = list(set(range(0, 4)) - {orig_idx, x_idx, y_idx})[0]

    ord_pts = np.array([points[orig_idx], points[x_idx], points[y_idx], points[xy_idx]])

    # If this is a square, sides must be almost identical
    th_square = 0.4
    d0 = point_dist(ord_pts[0], ord_pts[1])
    d1 = point_dist(ord_pts[0], ord_pts[2])
    d2 = point_dist(ord_pts[3], ord_pts[1])
    d3 = point_dist(ord_pts[3], ord_pts[2])

    c1 = abs(d0 / d1 - 1.0) < th_square
    c2 = abs(d1 / d2 - 1.0) < th_square
    c3 = abs(d2 / d3 - 1.0) < th_square

    if c1 and c2 and c3:
        return ord_pts
    else:
        return []


# todo: generalize the above idea to more complex patterns

def calib_square(points, K):
    K_1 = np.linalg.inv(K)

    if len(points) <= 0:
        return

    # Note the direction and order of points in pixels (rows, cols) and in image (x-y)
    dx = points[1] - points[0]
    dx_vec = np.array([dx[1], dx[0], 1.0]).reshape((3, 1))
    rx = K_1 @ dx_vec
    s = 5.0 / np.linalg.norm(rx)
    rx = s * rx / 5.0

    dy = points[2] - points[0]
    dy_vec = np.array([dy[1], dy[0], 1.0]).reshape((3, 1))
    ry = (s / 5.0) * K_1 @ dy_vec

    rz = np.cross(rx.reshape((1, 3)), ry.reshape((1, 3))).reshape((3, 1))

    R_cw = np.concatenate([rx, ry, rz], axis=1)
    rr = scipyR.from_matrix(R_cw)
    print(rr.as_rotvec())

    true_r = np.array([[-0.27446404],
                       [-0.10632083],
                       [-2.95396537]])
    print(true_r)

    t_cw = s * K_1 @ np.array([points[0, 1], points[0, 0], 1.0]).reshape((3, 1))


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.104:
                        squares.append(cnt)
    return squares


def test_squares(base_dir):

    file_name = os.path.join(base_dir, 'calib', 'images', 'image0.png')
    patch_name = os.path.join(base_dir, 'obj_detection', 'images', 'ball_patch.png')

    # load image
    im_color = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
    # load patch
    patch = cv2.imread(patch_name, cv2.IMREAD_GRAYSCALE)

    points = detect_black_square(image, patch)

    K = np.array([[625.24678685, 0., 297.50306391],
                  [0., 624.90293937, 251.99629151],
                  [0., 0., 1.]])
    calib_square(points, K)

    im_show = im_color
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
    for i in range(len(points)):
        im_show = cv2.circle(im_show, (int(points[i, 1]), int(points[i, 0])), 3, colors[i])

    # cv2.imshow('Image', im_show)
    # # cv2.imwrite(os.path.join(base_dir, 'patch0.png'), image[10:40, 540:580])
    #
    # cv2.waitKey(0)

    obj_center = calc_object_center(patch)

    im_show = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    im_show = cv2.drawMarker(im_show, np.int32(obj_center), (0, 0, 255), cv2.MARKER_CROSS, 3, 3)

    cv2.imshow('Image', im_show)
    cv2.waitKey(0)


def find_roi_thresh(th_img):
    mask = th_img > 0
    img_size = th_img.shape[:2]
    x_min = 0
    x_max = img_size[0]
    y_min = 0
    y_max = img_size[1]

    if not mask.any():
        return np.array([x_min, x_max, y_min, y_max])

    X, Y = get_mesh(img_size)
    pt_x = X[mask]
    pt_y = Y[mask]

    if len(pt_x) > 0:
        x_min = pt_x[np.argmin(pt_x)]
        x_max = pt_x[np.argmax(pt_x)]
        y_min = pt_y[np.argmin(pt_y)]
        y_max = pt_y[np.argmax(pt_y)]

    return np.array([x_min, y_min, x_max, y_max])


# for each of the contours detected, the shape of the contours is approximated using approxPolyDP()
# function and the contours are drawn in the image using drawContours() function
def detect_objects(contours, blank):
    font_scale = 0.5
    f_thick = 1
    for count in contours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        cv2.drawContours(blank, [approximations], 0, (0, 255, 0), 3)
        # the name of the detected shapes are written on the image
        i, j = approximations[0][0]
        if len(approximations) == 3:
            cv2.putText(blank, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 4:
            cv2.putText(blank, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 5:
            cv2.putText(blank, "Pentagon", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif 6 < len(approximations) < 15:
            cv2.putText(blank, "Ellipse", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        else:
            cv2.putText(blank, "Circle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        # displaying the resulting image as the output on the screen
        cv2.imshow("Resulting_image", blank)
        cv2.waitKey(0)


def track_hough_circles(img, last_img):
    # smooth both images
    img_s = smooth(img)
    last_img_s = smooth(last_img)

    # subtract images to find ROI
    sub_img = cv2.addWeighted(img_s, 0.5, last_img_s, -0.5, 255)
    # cv2.normalize(sub_img, sub_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(sub_img, None)
    ret, thresh = cv2.threshold(sub_img, 230, 255, cv2.THRESH_BINARY_INV)

    # find ROI
    roi = find_roi_thresh(thresh)
    center_roi = [(roi[0] + roi[2]) * 0.5, (roi[1] + roi[3]) * 0.5]

    # find circles
    circles = find_circles(img_s, hparams=(8, 60, 30, 10, 70))

    if circles is not None:
        if circles.shape[1] != 1:
            circles = circles.squeeze()
        else:
            circles = circles[0]
        dist_circles = []
        for circle in circles:
            dist_circles.append(point_dist(center_roi, circle[:2]))
        min_dist = np.argmin(dist_circles)

        return circles[min_dist]
    else:
        return np.array([center_roi[1], center_roi[0], 10])


def test_hough_circles(base_dir):

    video_file = os.path.join(base_dir, 'obj_detection', 'images', 'output.avi')
    another_video = os.path.join(base_dir, '..', '20231103_084334.mp4')
    scale = 0.5
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video file")

    last_frame = []

    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        if ret:
            if len(last_frame) <= 0:
                last_frame = frame
                sub_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                circle = track_hough_circles(frame, last_frame)
                last_frame = frame

                circle = np.int32(circle)
                frame = cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_watershed(img):
    img = img_resize(img, 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv.imshow('Thresh', thresh)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # cv.imshow("Sure_fg", unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv2.imshow('final', img)

    cv2.waitKey(0)


if __name__ == "__main__":
    # main_lines(sys.argv[1:])
    # main_circle(sys.argv[1:])

    data_dir = os.getenv('DATA_PATH')
    print(data_dir)

    img_path = os.path.join(data_dir, 'image20.png')

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    circles = find_circles(img, (8, 60, 30, 1, 30))

    img_show = img
    if circles is not None:
        if len(circles) == 1 and len(circles[0]) == 1:
            circles = circles[0]
        else:
            circles = circles.squeeze()
        for circle in circles:
            img_show = cv2.circle(img_show, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 2)

    cv2.imshow("Image", img_show[:, :, 1])

    cv2.waitKey(0)

    base_dir = os.getenv('DATA_PATH')
    # test_squares(base_dir)
    # test_hough_circles(base_dir)
    # test_klt_of(base_dir)

    # main_hough("")
    # main_april()

    settings_file = 'config/IAUN.yaml'
    settings = fst.load_settings(settings_file)

    image_file = 'assets/fira-signs.png'
    image_file1 = 'assets/fira-signs-1.png'
    image_file2 = 'assets/sudoku.png'
    image_file3 = 'assets/traffic-signs.jpg'
    image_file4 = 'assets/sample-contrast.jpg'

    image_folder = os.getenv('PATH_SEQ')

    image = cv2.imread(image_file4)
    #image = cvu.img_resize(image, 0.25)

    detector = ObjectDetection(settings)
    # cam = MultiCam(settings)

    t_basic_use(detector, image)
    # t_ml_video(detector, 'assets/project_video.mp4')
    # t_april_stream(settings, detector, image_folder)
    # t_lines(settings, detector, image_folder)
    # t_april_video(settings, detector, cam)

    cv2.destroyAllWindows()

    # main_hough("")
    # main_april()

    settings_file = 'config/IAUN.yaml'
    settings = fst.load_settings(settings_file)

    # image_file = 'assets/fira-signs.png'
    # image_file1 = 'assets/fira-signs-1.png'
    # image_file2 = 'assets/sudoku.png'
    # image_file3 = 'assets/traffic-signs.jpg'

    # image_folder = os.getenv('PATH_SEQ')

    # image = cv2.imread(image_file1)
    # image = cvu.img_resize(image, 0.25)

    detector = ObjectDetection(settings)

    # t_basic_use(detector, image)
    # t_ml_video(detector, 'assets/video9.avi')
    # t_detect_gates_ml(detector, 'assets/video9.avi')
    # t_april_stream(settings, detector, image_folder)
    # t_lines(settings, detector, image_folder)
    # t_april_video(detector, 'assets/video0.avi')
    # t_gate_detector(detector, 'assets/video9.avi')
    # t_color_detection('assets/video9.avi')

    cv2.destroyAllWindows()
