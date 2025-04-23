"""
    Draw Machine Learning outputs (mostly bounding boxes with labels and scores)
"""

import numpy as np
import cv2
from pathlib import Path
import yaml


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class ML_Drawer:
    def __init__(self):
        print('ML Drawer')

    @staticmethod
    def draw_detection(image, detections):

        if len(detections) > 0:
            for detection in detections:
                object_name, score, xmin, ymin, xmax, ymax = detection

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                # Example: 'person: 72%'
                label = '%s: %d%%' % (object_name, int(score * 100))
                # Get font size
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                # Draw label text
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 2)
        return image

    @staticmethod
    def draw_detections(image, detections):

        if detections is None or detections.results is None:
            return image

        for detection in detections.results:
            xmin, ymin, xmax, ymax = np.int32(detection.bbox)
            object_name = detection.name
            score = detection.conf

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            # Example: 'person: 72%'
            label = '%s: %d%%' % (object_name, int(score * 100))
            # Get font size
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            # Draw white box to put label text in
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            # Draw label text
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

        return image

    @staticmethod
    def draw_label(input_image, label, left, top):
        """Draw text onto image at location."""

        # Text parameters.
        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.7
        THICKNESS = 1

        # Colors
        BLACK = (0, 0, 0)
        BLUE = (255, 178, 50)
        YELLOW = (0, 255, 255)
        RED = (0, 0, 255)

        # Get text size.
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
        # Display text inside the rectangle.
        cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

    @staticmethod
    def write_detection():

        # Get filenames and paths
        image_fn = os.path.basename(image_path)
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn + '.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath, 'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (
                    detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    @staticmethod
    def draw_rect(image, bbox):
        # draw a rectangle on the image
        return cv2.rectangle(image, bbox[:2], bbox[2:], (255, 0, 255), 2)

    @staticmethod
    def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
        # Plots one bounding box on image 'im' using OpenCV
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def process_detections_tflite_yolo(pred, path, im0s, save_dir, names, img,
                                       save_img=True, hide_labels=False, hide_conf=False, line_thickness=2):
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            p = Path(p)  # to Path
            save_dir = Path(save_dir)
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[1:]  # print string    1,3,320,320

            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        ML_Drawer.plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                               line_thickness=line_thickness)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
        return im0s

