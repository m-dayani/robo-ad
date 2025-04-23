"""
    What is a drivable area and lane?
        1. Usually dark color (usually asphalt material)
        2. Planar area *
        3. There's a noticeable contrast between lane and its background (if visible) *
        4. Lanes are usually white or yellow *
        5. Lanes follow a pattern and perspective rules *
        6. Lanes are planar curvatures in general and lines in some cases *
        7. Lanes might have a thickness
        8. Smaller middle lines (disconnected pattern) have smaller length
        9. Lanes might be noisy (lost colors, occlusion, hard shadows and shining, ...)
"""
import argparse
import math

import numpy as np
import cv2
import sys
import os

sys.path.append('../../')
sys.path.append('../../data_loader/')
import cv_mine.cv_utils as cvu
from my_geo.observations import LaneLine
import mviz.viz_auto_driving as vizt
import tools.utils as fst
from data_loader.image_loader import ImageLoader


class LaneTracker:

    def __init__(self, settings, ferrari_path):

        # tracking mode: polyfit, contours, contours2
        self.tracking_mode = settings['tracking_mode']

        intrinsics = settings['intrinsics']
        self.K = np.array([[intrinsics[0], 0.0, intrinsics[2]],
                           [0.0, intrinsics[1], intrinsics[3]],
                           [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array(settings['distortion'])
        self.M = np.array(settings['perspective_transform']).reshape((3, 3))
        self.Minv = np.array(settings['perspective_trans_inv']).reshape((3, 3))
        self.img_size = np.array(settings['img_size'])[:2]
        self.crop_size = np.array(settings['crop_size'])
        self.filter_ksize = settings['filter_ksize']
        self.lim_img_sz = settings['lim_img_sz']
        self.grad_tresh = settings['grad_thresholds']
        self.hsl_tresh = settings['hsl_thresholds']
        self.warped_size = settings['warped_size']
        self.ferrari_path = ferrari_path

        self.line_r = LaneLine()
        self.line_l = LaneLine()

        self.last_center_rl = None
        self.last_center_ml = None
        self.lane_dist_px = np.int32(245)

        _, _, self.img_mask = cvu.get_mesh((self.img_size[1], self.img_size[0]))
        self.warped_x = np.linspace(0, self.warped_size[0]-1, self.warped_size[0], dtype=np.int32)

        self.morph_kernel = np.ones((5, 5), np.uint8)

        self.steering_scale = settings['steering_scale']

        print('LaneTracker (classic) initialized successfully in ' + self.tracking_mode + ' mode')

    def process(self, img_raw, vis=False):
        if self.tracking_mode == 'polyfit':
            return self.process_polyfit(img_raw, vis=vis)
        elif self.tracking_mode == 'contours':
            return self.process_contours(img_raw, vis=vis)
        elif self.tracking_mode == 'contours2':
            return self.process_contours2(img_raw, vis=vis)
        elif self.tracking_mode == 'contours3':
            return self.process_contours3(img_raw, vis=vis)

    def preprocess(self, img_raw):
        # Correcting for Distortion
        img_ud = cv2.undistort(img_raw, self.K, self.dist_coeffs)
        rows, cols = img_ud.shape[:2]

        # resize the image
        img_hf = img_ud
        if cols > self.lim_img_sz:
            img_hf = cvu.img_resize(img_ud, self.lim_img_sz / cols)

        # crop
        crop_dim = cvu.compute_crop(img_hf.shape, self.crop_size)
        img_crop = cvu.img_crop(img_hf, crop_dim)

        # color transform: isolate the best response color
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

        # denoising
        img_smooth = cv2.medianBlur(img_gray, self.filter_ksize)

        return img_smooth, img_crop, img_hf

    def process_polyfit(self, img_raw, vis=False):

        img_smooth, img_crop, img_hf = self.preprocess(img_raw)
        # return img_smooth

        # gradients and thresholding
        # todo: these operations still contain a lot of free-running params
        grad_th = self.grad_tresh
        img_comb_grad = cvu.get_combined_gradients(img_smooth, grad_th[:2], grad_th[2:4], grad_th[4:6], grad_th[6:])
        # return img_comb_grad

        hsl_th = self.hsl_tresh
        img_comb_hsl = cvu.get_combined_hls(img_crop, hsl_th[:2], hsl_th[2:4], hsl_th[4:])
        # h, s, l = cv2.split(cv2.cvtColor(img_crop, cv2.COLOR_BGR2HLS))
        # return img_comb_hsl

        # combine thresh
        img_combined = cvu.combine_grad_hls(img_comb_grad, img_comb_hsl)
        # img_combined = img_comb_grad
        # return img_combined

        # morphology
        img_dilate = cv2.dilate(img_combined, (5, 5), iterations=5)
        img_eroded = cv2.erode(img_dilate, (5, 5), iterations=3)
        img_morph = img_eroded
        # return img_morph

        # perspective warp
        img_warp = cv2.warpPerspective(img_morph, self.M, self.warped_size, flags=cv2.INTER_LINEAR)
        # return img_warp

        # search lines
        img_searching = self.get_lane_lines_img(img_warp, vis=vis)
        # return img_searching

        # todo: and even more:
        # 1. stereo triangulation and distance estimation
        # 2. using rulebook relations (distance between lanes, ...)

        # visualization (optional)
        if vis:
            return vizt.illustrate_final(img_hf, img_searching, self.line_l, self.line_r, self.Minv,
                                         img_combined.shape[:2], self.crop_size, self.ferrari_path)
        return img_searching

    def contour_analysis(self, contour, img_bin, last_center):
        # length of contour
        n_pts = len(contour)
        contour_center = np.mean(contour, axis=0)

        # distance of contour center with last center
        last_center_dist = 0.001
        if last_center is None:
            last_center = contour_center
        else:
            last_center_dist += cvu.point_dist(last_center, contour_center)
        last_center_score = 1.0 / last_center_dist

        # distance of contour center with expected location
        hb, wb = img_bin.shape[:2]
        expected_loc = np.array([wb * 0.75, hb * 0.5], dtype=np.int32)
        expected_loc_dist = cvu.point_dist(expected_loc, contour_center) + 0.001
        expected_loc_score = 1.0 / expected_loc_dist

        # row distance (lane width)
        min_row = np.min(contour[:, 1])
        max_row = np.max(contour[:, 1])
        contour_length = max_row - min_row
        test_rows = np.linspace(min_row, max_row, 10, dtype=np.int32)
        row_dists = np.zeros(len(test_rows)) - 1
        for i, test_row in enumerate(test_rows):
            cnt_rows = contour[contour[:, 1] == test_row]
            if len(cnt_rows) > 0:
                search_x = cnt_rows[0, 0]
                xw = self.warped_x[img_bin[test_row, :] > 0]
                xwc = np.bitwise_and(xw > search_x - 30, xw < search_x + 30)
                xw = xw[xwc]
                dist_mat = abs(xw - xw.reshape(-1,1))
                try:
                    search_mat = dist_mat[dist_mat > 0]
                    if len(search_mat) > 0:
                        row_dists[i] = np.max(search_mat)
                except ValueError:
                    print('contour_analysis, ValueError')
        avg_row_dist = 0
        valid_row_dists = row_dists[row_dists > 0]
        if len(valid_row_dists) > 0:
            avg_row_dist = np.mean(valid_row_dists)
        row_dist_score = 1.0 / (abs(avg_row_dist - 20) + 0.001)

        return [n_pts, last_center_score, row_dist_score, contour_length, expected_loc_score, avg_row_dist], last_center

    def unwarp_contour(self, contour, search_win, img_bin, refine_pts=True):
        # this neighborhood retrival is still time-consuming and a bottleneck
        # the down-sampled version is much faster
        contour_ds = contour
        if len(contour) > 2 * search_win:
            # downsample long contours
            contour_ds = contour[::int(search_win / 3)]
        c_ds_homo = np.float32(contour_ds).reshape(-1, 1, 2)
        c_img_pts = cv2.perspectiveTransform(c_ds_homo, self.Minv)
        c_img_pts = np.int32(np.squeeze(c_img_pts))
        if len(c_img_pts.shape) < 2:
            c_img_pts = c_img_pts.reshape(-1, 2)
        if not refine_pts:
            return c_img_pts
        # refine the contour by searching neighborhood
        all_points, mask = cvu.get_neighbors_v(c_img_pts, search_win, img_bin.shape[:2])
        valid_pts = all_points[mask]
        if len(valid_pts) <= 0:
            return c_img_pts
        valid_pts = np.unique(valid_pts, axis=0)
        X = valid_pts[:, 0]
        Y = valid_pts[:, 1]
        c_bin_vals = img_bin[Y, X] > 0
        if np.count_nonzero(c_bin_vals) <= 0:
            return c_img_pts
        xy_refined = np.concatenate((X[c_bin_vals].reshape(-1, 1), Y[c_bin_vals].reshape(-1, 1)), axis=1)
        len_ratio = len(xy_refined) / len(c_img_pts)
        if len_ratio > 2:
            xy_refined = xy_refined[::int(len_ratio)]
        return xy_refined

    def estimate_middle_lane(self, right_lane, search_win, img_bin, refine_pts=False):
        middle_lane = np.copy(right_lane)
        # this '240' highly depends on the perspective warp sizes and lane sizes
        # (varies between 240~250 in this case)
        middle_lane[:, 0] -= self.lane_dist_px
        ml_unwarped = self.unwarp_contour(middle_lane, search_win, img_bin, refine_pts)
        return middle_lane, ml_unwarped

    def process_contours(self, img_raw, vis=False):
        img_show = np.copy(img_raw)
        right_lane, middle_lane, steering_ang, confidence = [], [], 0, 0
        img_smooth, img_crop, img_hf = self.preprocess(img_raw)

        # detect canny edges
        img_canny = cv2.Canny(img_smooth, 120, 240)

        # optional: morphology
        img_morph = img_canny

        # warp canny binary image: bird-eye view
        # detect contours in the reduced space and extract features from perspective image
        img_warp_bin = cv2.warpPerspective(img_morph, self.M, self.warped_size, flags=cv2.INTER_LINEAR)

        # detect contours
        hc = int(img_warp_bin.shape[1]/2)
        contours, h = cv2.findContours(img_warp_bin[:, hc:],
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return img_show, right_lane, middle_lane, steering_ang, confidence

        # sort contours by length
        # contours = sorted(contours, key=lambda x: len(x), reverse=True)

        # contour analysis
        n_contours = len(contours)
        contours_info = np.zeros((n_contours, 6))
        contour_center = []
        contours_scored = []
        for i, contour in enumerate(contours):
            contour = np.squeeze(contour)
            if len(contour.shape) < 2:
                contour = contour.reshape(1, 2)
            contour[:, 0] += hc
            contour_score, last_center = self.contour_analysis(contour, img_warp_bin, self.last_center_rl)
            contours_info[i, :] = contour_score
            contour_center.append(last_center)
            contours_scored.append(contour)

        # select the maximum contour as the chosen right lane
        contour_scores = np.sum(contours_info[:, :4], axis=1)
        cs_idx = np.argmax(contour_scores)
        right_lane = contours_scored[cs_idx]
        self.last_center_rl = contour_center[cs_idx]
        contour_score = contours_info[cs_idx]
        if contours_info[cs_idx, 0] < 150:
            return img_show, right_lane, middle_lane, steering_ang, confidence

        # warped to perspective contour conversion
        win_size = 41
        rl_unwarped = self.unwarp_contour(right_lane, 21, img_canny, False)

        # the middle lane is always at the same distance from the main lane in the bird-eye view
        middle_lane, ml_unwarped = self.estimate_middle_lane(right_lane, 21, img_canny, False)

        # color segmentation along the longest contour
        labels_sqz, cls0_pts, cls0_ft, cls1_pts, cls1_ft, color_score = (
            self.color_analysis(img_crop, img_smooth, rl_unwarped, win_size))
        # labels_sqz_ml, cls0_pts_ml, cls0_ft_ml, cls1_pts_ml, cls1_ft_ml, color_score_ml = (
        #     self.color_analysis(img_crop, img_smooth, ml_unwarped, win_size))

        # output an uncertainty score based on the presence of the other lane, colors, ...
        # problem 1: false positives
        # problem 2: bad color segmentation due to HDR
        score_vec = np.hstack((contour_score, color_score))
        # print(score_vec)
        confidence = self.calculate_lane_confidence(score_vec)

        # calculate and update the steering angle based on the contour start and end x location
        steering_ang = self.calculate_steering_angle(right_lane)

        if vis and confidence > 60:
            if len(cls0_pts) > 0:
                img_show[cls0_pts[:, 1], cls0_pts[:, 0]] = (255, 0, 0)
            if len(cls1_pts) > 0:
                img_show[cls1_pts[:, 1], cls1_pts[:, 0]] = (0, 0, 255)
            # if len(cls0_pts_ml) > 0:
            #     img_show[cls0_pts_ml[:, 1], cls0_pts_ml[:, 0]] = (255, 0, 0)
            # if len(cls1_pts_ml) > 0:
            #     img_show[cls1_pts_ml[:, 1], cls1_pts_ml[:, 0]] = (0, 0, 255)
            img_show = cv2.drawContours(img_show, [rl_unwarped], -1, (0, 255, 0), 2)
            img_show = cv2.drawContours(img_show, [ml_unwarped], -1, (0, 255, 0), 2)
            img_show = cv2.putText(img_show, f'Confidence: %.0f' % confidence, (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            img_show = cv2.putText(img_show, f'Steering Angle: %.2f' % steering_ang, (50, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        return img_show, right_lane, middle_lane, steering_ang, confidence

    def process_contours3(self, img_raw, vis=False):

        th_len_contour = 80

        img_show = np.copy(img_raw)
        steering_ang = -1.0
        img_smooth, img_crop, img_hf = self.preprocess(img_raw)

        # detect canny edges
        img_canny = cv2.Canny(img_smooth, 120, 240)

        # optional: morphology
        img_morph = img_canny

        # warp canny binary image: bird-eye view
        # detect contours in the reduced space and extract features from perspective image
        img_warp_bin = cv2.warpPerspective(img_morph, self.M, self.warped_size, flags=cv2.INTER_LINEAR)
        img_warp_bin[img_warp_bin > 0] = 255

        hiw, wiw = img_warp_bin.shape[:2]
        wiwh = 0.5 * wiw
        img_wleft = np.copy(img_warp_bin)
        img_wleft[:, int(wiwh):] = 0
        left_contour = img_wleft.shape[1] - 1 - np.argmax(img_wleft[:,::-1], axis=1)
        img_wright = np.copy(img_warp_bin)
        img_wright[:, :int(wiwh)] = 0
        right_contour = np.argmax(img_wright, axis=1)

        img_c = np.zeros_like(img_warp_bin)

        rows = np.arange(img_c.shape[0])
        img_c[rows, left_contour] = 255
        img_c[rows, right_contour] = 255

        img_dilation = cv2.dilate(img_c, self.morph_kernel, iterations = 1)
        img_erosion = cv2.erode(img_dilation, self.morph_kernel, iterations=1)

        # detect contours
        contours, h = cv2.findContours(img_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return img_show, steering_ang

        # sort contours by length
        contours = sorted(contours, key=lambda x: len(x), reverse=True)

        steering_angles = []
        contours_length = []
        for contour in contours:
            contour = contour.squeeze().reshape(-1, 2)
            if len(contour) > th_len_contour:
                min_y_idx = np.argmin(contour[:, 1])
                max_y_idx = np.argmax(contour[:, 1])
                pt2 = contour[min_y_idx, :]
                pt1 = contour[max_y_idx, :]
                pt_diff = pt2 - pt1
                ang = np.arctan2(pt_diff[1], pt_diff[0]) * 180.0 / np.pi
                if ang < 0:
                    ang += 180
                steering_angles.append(ang)
                contours_length.append(len(contour))

        if len(steering_angles) > 0:
            # steering_angles = sorted(steering_angles)
            # steering_ang = np.mean(steering_angles)
            # compute a weighted mean
            contours_length = np.float32(contours_length)
            contours_length /= max(contours_length)
            sum_w = np.sum(contours_length)
            steering_angles = np.float32(steering_angles)
            steering_ang = np.sum(steering_angles * contours_length) / sum_w
            steering_ang_strength = steering_ang - 90.0
            steering_ang = steering_ang_strength * self.steering_scale + 90.0

        # for contour in contours:
        if vis:
            img_show = cv2.cvtColor(img_warp_bin, cv2.COLOR_GRAY2BGR)
            for contour in contours:
                if len(contour) > th_len_contour:
                    img_show = cv2.drawContours(img_show, [contour], -1, (0, 0, 255), 2)

        return img_show, steering_ang

    def measure_curvature(self):
        """
        #---------------------
        # This function measures curvature of the left and right lane lines
        # in radians.
        # This function is based on code provided in curvature measurement lecture.
        #
        """

        left_lane = self.line_l
        right_lane = self.line_r
        img_size = self.img_size

        ploty = left_lane.ally

        leftx, rightx = left_lane.allx, right_lane.allx

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # U.S. regulations that require a  minimum lane width of 12 feet or 3.7 meters,
        # and the dashed lane lines are 10 feet or 3 meters long each.
        # >> http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC

        # Below is the calculation of radius of curvature after correcting for scale in x and y
        # Define conversions in x and y from pixels space to meters
        lane_width = abs(right_lane.startx - left_lane.startx)
        if lane_width == 0:
            lane_width = 200
        # todo: hardwired values
        ym_per_pix = 30 / img_size[1]  # meters per pixel in y dimension
        xm_per_pix = 3.7 * (img_size[1] / img_size[0]) / lane_width  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # radius of curvature result
        left_lane.radius_of_curvature = left_curverad
        right_lane.radius_of_curvature = right_curverad

    def line_search_reset(self, binary_img, vis=False):
        """
        #---------------------
        # After applying calibration, thresholding, and a perspective transform to a road image,
        # I have a binary image where the lane lines stand out clearly.
        # However, I still need to decide explicitly which pixels are part of the lines
        # and which belong to the left line and which belong to the right line.
        #
        # This lane line search is done using histogram and sliding window
        #
        # The sliding window implementation is based on lecture videos.
        #
        # This function searches lines from scratch, i.e. without using info from previous lines.
        # However, the search is not entirely a blind search, since I am using histogram information.
        #
        # Use Cases:
        #    - Use this function on the first frame
        #    - Use when lines are lost or not detected in previous frames
        #
        """

        left_lane = self.line_l
        right_line = self.line_r

        # I first take a histogram along all the columns in the lower half of the image
        histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)

        # if vis:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int32(histogram.shape[0] / 2)
        leftX_base = np.argmax(histogram[:midpoint])
        rightX_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        num_windows = 9

        # Set height of windows
        window_height = np.int32(binary_img.shape[0] / num_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        current_leftX = leftX_base
        current_rightX = rightX_base

        # Set minimum number of pixels found to recenter window
        min_num_pixel = 50

        # Create empty lists to receive left and right lane pixel indices
        win_left_lane = []
        win_right_lane = []

        window_margin = left_lane.window_margin

        # Step through the windows one by one
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_img.shape[0] - (window + 1) * window_height
            win_y_high = binary_img.shape[0] - window * window_height
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

            if vis:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                    nonzerox <= win_leftx_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                    nonzerox <= win_rightx_max)).nonzero()[0]
            # Append these indices to the lists
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(left_window_inds) > min_num_pixel:
                current_leftX = np.int32(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = np.int32(np.mean(nonzerox[right_window_inds]))

        # Concatenate the arrays of indices
        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        # Extract left and right line pixel positions
        leftx = nonzerox[win_left_lane]
        lefty = nonzeroy[win_left_lane]
        rightx = nonzerox[win_right_lane]
        righty = nonzeroy[win_right_lane]

        if vis:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        if len(leftx) <= 0 or len(lefty) <= 0 or len(rightx) <= 0 or len(righty) <= 0:
            return out_img

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_lane.current_fit = left_fit
        right_line.current_fit = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        # ax^2 + bx + c
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_lane.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        if len(left_lane.prevx) > 10:
            left_avg_line = self.smoothing(left_lane.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_lane.current_fit = left_avg_fit
            left_lane.allx, left_lane.ally = left_fit_plotx, ploty
        else:
            left_lane.current_fit = left_fit
            left_lane.allx, left_lane.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:
            right_avg_line = self.smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        left_lane.startx, right_line.startx = left_lane.allx[len(left_lane.allx) - 1], right_line.allx[
            len(right_line.allx) - 1]
        left_lane.endx, right_line.endx = left_lane.allx[0], right_line.allx[0]

        # Set detected=True for both lines
        left_lane.detected, right_line.detected = True, True

        self.measure_curvature()

        return out_img

    def line_search_tracking(self, b_img, vis=False):
        """
        #---------------------
        # This function is similar to `line_seach_reset` function, however, this function utilizes
        # the history of previously detcted lines, which is being tracked in an object of Line class.
        #
        # Once we know where the lines are, in previous frames, we don't need to do a blind search, but
        # we can just search in a window_margin around the previous line position.
        #
        # Use Case:
        #    - Highly targetted search for lines, based on info from previous frame
        #
        """

        left_line = self.line_l
        right_line = self.line_r
        img_size = self.img_size

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((b_img, b_img, b_img)) * 255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = b_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Get margin of windows from Line class. Adjust this number.
        window_margin = left_line.window_margin

        left_line_fit = left_line.current_fit
        right_line_fit = right_line.current_fit
        leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
        leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
        rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[
            2] - window_margin
        rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[
            2] + window_margin

        # Identify the nonzero pixels in x and y within the window
        left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
        right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

        if vis:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        if lefty is None or leftx is None or righty is None or rightx is None or \
                len(lefty) <= 0 or len(leftx) <= 0 or len(righty) <= 0 or len(rightx) <= 0:
            return out_img

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

        # ax^2 + bx + c
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        leftx_avg = np.average(left_plotx)
        rightx_avg = np.average(right_plotx)

        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        if len(left_line.prevx) > 10:  # take at least 10 previously detected lane lines for reliable average
            left_avg_line = self.smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:  # take at least 10 previously detected lane lines for reliable average
            right_avg_line = self.smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # Compute Standard Deviation of the distance between X positions of pixels of left and right lines
        # If this STDDEV is too high, then we need to reset our line search, using line_search_reset
        stddev = np.std(right_line.allx - left_line.allx)

        if stddev > 80:
            left_line.detected = False

        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[
            len(right_line.allx) - 1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        self.measure_curvature()

        return out_img

    def get_lane_lines_img(self, binary_img, vis=False):
        """
        #---------------------
        # This function finds left and right lane lines and isolates them.
        # If first frame or detected==False, it uses line_search_reset,
        # else it tracks/finds lines using history of previously detected lines, with line_search_tracking
        #
        """

        if not self.line_l.detected:
            return self.line_search_reset(binary_img, vis)
        else:
            return self.line_search_tracking(binary_img, vis)

    def process_contours2(self, img_raw, vis=False):
        # img_h, img_w = img_raw.shape[:2]
        # img_show = np.copy(img_raw)

        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        img_smooth = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)

        # Correcting for Distortion
        img_ud = cv2.undistort(img_smooth, self.K, self.dist_coeffs)

        img_canny = cv2.Canny(img_ud, 120, 240)

        img_gy = cv2.Sobel(img_canny, cv2.CV_32F, 1, 0, ksize=3)
        img_gy[img_gy > 0] = 255
        img_gy = np.uint8(img_gy)
        # img_gy = img_canny

        # + morphology??
        # img_dilation = cv2.dilate(img_gy, self.morph_kernel, iterations=1)
        # img_closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, self.morph_kernel)
        # img_erosion = cv2.erode(img_closing, self.morph_kernel, iterations=1)

        img_warp_bin = cv2.warpPerspective(img_gy, self.M, self.warped_size, flags=cv2.INTER_LINEAR)

        lines = cv2.HoughLines(img_warp_bin, 1, np.pi / 180, 150, None, 0, 0)

        img_show = cv2.cvtColor(img_warp_bin, cv2.COLOR_GRAY2BGR)

        # process lines to find the steering angle
        line_info = dict()
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = np.array([x0 + 1000 * (-b), y0 + 1000 * a])
                pt2 = np.array([x0 - 1000 * (-b), y0 - 1000 * a])

                line_center = 0.5 * (pt1 + pt2)
                dp = pt2 - pt1
                line_angle = np.arctan2(dp[1], dp[0]) * 180 / np.pi
                if line_angle < 0:
                    line_angle += 180
                line_info[line_angle] = line_center

                if vis:
                    cv2.line(img_show, np.int32(pt1), np.int32(pt2), (0, 255, 255), 3, cv2.LINE_AA)

        line_angles = sorted(line_info.keys())
        med_angle = 87
        if len(line_angles) > 0:
            # med_angle = line_angles[int(len(line_angles) / 2)]
            med_angle = np.mean(line_angles)
        # turn the other way to compensate for misaligned direction
        med_angle = 180.0 - med_angle

        return img_show, med_angle

    def get_bird_eye_gray(self, img_input):
        img_smooth, _, _ = self.preprocess(img_input)

        img_bird_eye = cv2.warpPerspective(img_smooth, self.M, self.warped_size, flags=cv2.INTER_LINEAR)
        return img_bird_eye

    @staticmethod
    def color_analysis(img_color, img_gray, contour, search_win):

        color_ang_score, color_abs_score, poly_score = (0, 0, 0)
        all_points, mask = cvu.get_neighbors_v(contour, search_win, img_color.shape[:2])
        valid_pts = all_points[mask]
        if len(valid_pts) <= 0:
            return [], [], [], [], [], [color_ang_score, color_abs_score, poly_score]

        valid_pts = np.unique(valid_pts, axis=0)
        X = valid_pts[:, 0]
        Y = valid_pts[:, 1]

        # based on this feature analysis decide if the main lane is found
        ft_color = img_color[Y, X]
        ft_gray = img_gray[Y, X].reshape((-1, 1))
        ft_space = np.concatenate((ft_color, ft_gray), axis=1)

        kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        retval, labels, centers = cv2.kmeans(np.float32(ft_space), 2,
                                             None, kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels_sqz = np.squeeze(labels)
        cls0_idx = 0
        if np.mean(ft_space[labels_sqz == 0]) > np.mean(ft_space[labels_sqz == 1]):
            cls0_idx = 1
        cls0_pts = valid_pts[labels_sqz == cls0_idx]
        cls1_pts = valid_pts[labels_sqz == int(1 - cls0_idx)]
        cls0_ft = ft_space[labels_sqz == cls0_idx]
        cls1_ft = ft_space[labels_sqz == int(1 - cls0_idx)]

        if len(cls0_pts) <= 10 or len(cls1_pts) <= 10:
            return labels_sqz, cls0_pts, cls0_ft, cls1_pts, cls1_ft, [color_ang_score, color_abs_score, poly_score]

        cls0_mean = np.mean(cls0_ft, axis=0)
        cls1_mean = np.mean(cls1_ft, axis=0)
        cls0_abs = np.linalg.norm(cls0_mean)
        cls0_norm = cls0_mean / (cls0_abs + 0.001)
        cls1_abs = np.linalg.norm(cls1_mean)
        cls1_norm = cls1_mean / (cls1_abs + 0.001)
        cs_theta = np.sum(cls0_norm * cls1_norm)
        # theta = np.arccos(cs_theta)

        # in the RGB domain, lane color vectors are aligned and one is substantially smaller than the other
        color_ang_score = cs_theta
        color_abs_score = abs(max(cls0_abs, cls1_abs) / (min(cls0_abs, cls1_abs) + 0.001))

        # can fit a cure to the color points and see if all curves match
        cf_contour, cm_contour = np.polyfit(contour[:, 1], contour[:, 0], 3, cov=True)
        cf_cls1, cm_cls1 = np.polyfit(cls1_pts[:, 1], cls1_pts[:, 0], 3, cov=True)
        err_poly = np.linalg.norm(cf_contour - cf_cls1) + 0.001
        poly_score = 1.0 / err_poly

        return labels_sqz, cls0_pts, cls0_ft, cls1_pts, cls1_ft, [color_ang_score, color_abs_score, poly_score]

    @staticmethod
    def calculate_lane_confidence(score_vec):
        """
        :param score_vec: [n_points, inv_last_dist, row_dist, c_length,
                           loc_score, row_dist, color_ang, color_abs, poly]
        """
        score = 0
        if score_vec[0] > 200:
            score += 1
        if score_vec[1] > 0.02:
            score += 1
        if score_vec[3] > 100:
            score += 1
        if 15 < score_vec[5] < 25:
            score += 1
        if score_vec[6] > 0.8:
            score += 1
        if score_vec[7] > 2:
            score += 1
        if score_vec[8] > 0.001:
            score += 1

        score = score / 8 * 100
        return score

    @staticmethod
    def calculate_steering_angle(lane):
        l_start = lane[np.argmax(lane[:, 1])]
        l_end = lane[np.argmin(lane[:, 1])]
        l_diff = l_end - l_start
        return np.arctan2(l_diff[1], l_diff[0]) / np.pi * 180 + 90

    @staticmethod
    def smoothing(lines, prev_n_lines=3):
        # collect lines & print average line
        """
        #---------------------
        # This function takes in lines, averages last n lines
        # and returns an average line
        #
        """
        lines = np.squeeze(lines)  # remove single dimensional entries from the shape of an array
        avg_line = np.zeros(720)
        initialized = False

        for i, line in enumerate(reversed(lines)):
            if i == prev_n_lines:
                break
            if not initialized:
                avg_line = np.zeros(len(line))
                initialized = True
            avg_line += line
        avg_line = avg_line / prev_n_lines

        return avg_line


class MyTrackBar:
    def __init__(self, num_tracks=8, window_name='Parameters'):

        self.frame = np.zeros((640, 5, 3))
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.num_tracks = num_tracks

        self.tracks = []
        for i in range(num_tracks):
            track_name = 'Param #' + str(i)
            self.tracks.append(track_name)
            cv2.createTrackbar(track_name, self.window_name, 0, 255, self.nothing)

    def nothing(self, x):
        pass

    def get_param(self, idx):

        track_name = 'Param #' + str(idx)
        if track_name in self.tracks:
            return cv2.getTrackbarPos(track_name, self.window_name)
        return 0

    def map(self, v, a0, b0, a1, b1):
        return (v - a0) / b0 * b1 + a1


def t_norm_op(img_loader, lane_tracker):
    # video = cv2.VideoWriter('assets/lane_finder_real.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (640, 480))

    cnt = 0
    while img_loader.is_ok:

        frame = img_loader.get_next()

        cnt += 1
        # if cnt < 200:
        #     continue

        if frame is None:
            break

        frame = frame.frame
        # frame = 255 - frame
        if lane_tracker.tracking_mode == 'contours':
            out_img, _, _, steering, uncertainty = lane_tracker.process_contours(frame, vis=True)
        elif lane_tracker.tracking_mode == 'contours2':
            out_img, steering = lane_tracker.process_contours2(frame, vis=True)
        elif lane_tracker.tracking_mode == 'contours3':
            out_img, steering = lane_tracker.process_contours3(frame, vis=True)
        else:
            out_img = lane_tracker.process_polyfit(frame, vis=True)
            steering = 0.0

        out_img = cv2.putText(out_img, str(steering), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Lane Finder", out_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        # video.write(out_img)

    img_loader.close()
    cv2.destroyAllWindows()


def t_params(img_loader, lane_tracker, my_trackbar):
    # init params
    for i, val in enumerate(lane_tracker.grad_tresh):
        track_name = 'Param #' + str(i)
        if i == 6 or i == 7:
            val = int((val - 0) / 2.0 * 255 + 0)
        cv2.setTrackbarPos(track_name, my_trackbar.window_name, val)

    while img_loader.is_ok:
        frame = img_loader.get_next()

        if frame is None:
            break

        frame = frame[:, :640]
        # frame = 255 - frame

        # update params
        grad_tresh = lane_tracker.grad_tresh
        for i in range(8):
            grad_tresh[i] = my_trackbar.get_param(i)
            if i == 6 or i == 7:
                grad_tresh[i] = (grad_tresh[i] - 0) / 255 * 2.0 - 0.0
        lane_tracker.grad_tresh = grad_tresh

        out_img = lane_tracker.process(frame, vis=True)
        img_canny = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 120, 240)
        contours, h = cv2.findContours(out_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: len(x), reverse=True)
        blank = cv2.merge((img_canny, img_canny, img_canny))

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for i in range(3):
            blank = cv2.drawContours(blank, [contours[i]], -1, colors[i], 2)

        cv2.imshow("Lane Finder", out_img)
        cv2.imshow("Canny", blank)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    img_loader.close()
    cv2.destroyAllWindows()


def t_bird_eye(img_loader, lane_tracker):

    while img_loader.is_ok:

        frame = img_loader.get_next()

        if frame is None:
            break

        frame = frame.frame
        out_img = lane_tracker.get_bird_eye_gray(frame)

        cv2.imshow("Bird-eye Image", out_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    img_loader.close()
    cv2.destroyAllWindows()

img_points = [[]]
def mouse_callback(event, x, y, flags, params):
    global img_points
    if event == 1:
        if params == 0:
            # print(len(img_points))
            img_points[-1].append([x, y])
    elif event == 2:
        # make a new points list
        img_points.append([])


img_stats_instructions = '''
Image Stats
    Left Click:         Select points
    Right+Left Click:   Select a new points list
    R-Key:              Reset all points
    S-Key:              Show image and points
    ESC:                Quit stats
-------------------------------------------------
'''
def t_stat(img_loader, lane_finder):
    print(img_stats_instructions)

    global img_points

    win_name = 'image0'
    window_width, window_height = (640, 480)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window_width, window_height)
    cv2.setMouseCallback(win_name, mouse_callback, 0)

    win_size = 41
    line_step = win_size / 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    grad_ksize = 3

    while img_loader.is_ok:

        frame = img_loader.get_next()

        if frame is not None:

            frame = frame[:, :int(frame.shape[1]/2)]
            img_show = np.copy(frame)

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_smooth = cv2.medianBlur(img_gray, 5)

            img_canny = cv2.Canny(img_smooth, 120, 240)

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # these use Gaussian smoothing internally, so don't need to be explicit
            img_laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            img_grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=grad_ksize)
            img_grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=grad_ksize)
            img_grad_mag = np.sqrt(img_grad_x ** 2 + img_grad_y ** 2)
            img_grad_ang = np.arctan2(np.absolute(img_grad_y), np.absolute(img_grad_x))

            img_warp = cv2.warpPerspective(img_canny, lane_finder.M, lane_finder.warped_size, flags=cv2.INTER_LINEAR)

            cv2.imshow(win_name, frame)
            cv2.imshow("Bird-eye View", cvu.img_resize(img_warp, 0.5))

            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break
            elif k == ord('r'):
                img_points = [[]]
            elif k == ord('s'):
                colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
                for i, pt_list in enumerate(img_points):

                    line_pts = cvu.get_line_points_from_segments(pt_list, line_step=line_step)

                    # get all points around the selected points
                    all_pts, mask = cvu.get_neighbors_v(line_pts, win_size, frame.shape[:2])
                    valid_pts = np.unique(all_pts[mask], axis=0)

                    # Print stats
                    print('Stats for point group #' + str(i))
                    gray_images = (img_smooth, img_grad_x, img_grad_y, img_grad_mag, img_grad_ang, img_laplacian)
                    gray_labels = ('gray', 'Gx', 'Gy', 'G_mag', 'G_ang', 'laplacian')
                    labels_sqz = cvu.print_img_stats(valid_pts, (frame, img_hsv), ('BGR', 'HSV'),
                                                     gray_images, gray_labels, criteria)

                    # Visualization:
                    # if len(contour) > 500:
                    cls0_pts = valid_pts[labels_sqz[0] == 0]
                    cls1_pts = valid_pts[labels_sqz[0] == 1]
                    img_show[cls0_pts[:, 1], cls0_pts[:, 0]] = (0, 255, 0)
                    img_show[cls1_pts[:, 1], cls1_pts[:, 0]] = (255, 0, 0)
                    # img_show[line_pts[:, 1], line_pts[:, 0]] = (0, 0, 255)
                    # cls0_pts = valid_pts[labels_sqz[1] == 0]
                    # cls1_pts = valid_pts[labels_sqz[1] == 1]
                    # img_show[cls0_pts[:, 1], cls0_pts[:, 0], 0] = 255
                    # img_show[cls1_pts[:, 1], cls1_pts[:, 0], 2] = 255
                    last_pt = None
                    for pt in pt_list:
                        if last_pt is not None:
                            img_show = cv2.line(img_show, last_pt, pt, colors[i % 3], 2)
                        last_pt = pt

                cv2.imshow(win_name, img_show)
                cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Test LaneFinder class: find and track lanes in input images
    ''')
    parser.add_argument('path_config', help='settings file path')
    parser.add_argument('img_src', help='image source')
    parser.add_argument('--load_mode', help='load mode (default: video)', default='video')
    parser.add_argument('--path_ferrari', help='ferrari.png path (default: ferrari.png)', default='ferrari.png')
    # parser.add_argument('--tracker_mode', help='Tracker mode (default: polyfit)', default='polyfit')
    args = parser.parse_args()

    settings_file = args.path_config
    settings = fst.load_settings(settings_file)

    ds_root = settings['ds_root']
    ferrari_path = os.path.join(ds_root, 'images', 'ferrari.png')
    lane_tracker = LaneTracker(settings, ferrari_path)
    # lane_tracker.tracking_mode = args.tracker_mode

    img_src = os.path.join(ds_root, args.img_src)
    img_loader = ImageLoader(img_src, args.load_mode)

    # my_trackbar = MyTrackBar(window_name='Lane Finder')

    t_norm_op(img_loader, lane_tracker)
    # t_params(img_loader, lane_tracker, my_trackbar)
    # t_stat(img_loader, lane_tracker)
    # t_bird_eye(img_loader, lane_tracker)
