import argparse

import numpy as np
import cv2
import os
import sys


def img_resize(img, scale):
    height, width = img.shape[:2]
    scaled_dim = (int(width * scale), int(height * scale))
    return cv2.resize(img, scaled_dim, cv2.INTER_AREA)


def img_resize1(img, w, h, ratio):
    img_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(img, img_size)


def point_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def calc_img_gradient(img, grad_ksize=3):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=grad_ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=grad_ksize)
    # mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0.0)
    m = np.sqrt(gx * gx + gy * gy)
    mag = np.uint8(cv2.normalize(m, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255)
    ang = np.arctan2(gx, gy)
    # cv2.imshow('Gradient Mag (linear)', mag)
    # cv2.imshow('Gradient Mag (rms)', mag1)
    return mag, ang, gx, gy


def color_thresh(b, g, r, th_vec):
    if b is None or g is None or r is None:
        return b

    result = np.zeros(b.shape[:2])

    if len(th_vec) != 6:
        return result

    # b, g, r = cv2.split(img)
    x1 = np.bitwise_and(th_vec[0] <= b, b <= th_vec[1])
    x2 = np.bitwise_and(th_vec[2] <= g, g <= th_vec[3])
    x3 = np.bitwise_and(th_vec[4] <= r, r <= th_vec[5])
    bg = np.bitwise_and(x1, x2)
    result = np.bitwise_and(bg, x3)

    return result


def check_color_thresh(color_vec, thresh_vec):
    if len(color_vec) != 3 or len(thresh_vec) != 6:
        return False
    return thresh_vec[0] < color_vec[0] < thresh_vec[1] and \
        thresh_vec[2] < color_vec[1] < thresh_vec[3] and \
        thresh_vec[4] < color_vec[2] < thresh_vec[5]


def is_in_image(point, image_size):
    return 0 <= point[0] < image_size[1] and 0 <= point[1] < image_size[0]


def clip_point_in_image(pt, img_sz):
    px = max(pt[0], 0)
    px = min(px, img_sz[1] - 1)
    py = max(pt[1], 0)
    py = min(py, img_sz[0] - 1)
    return px, py


def find_dominant_colors(img, n_segments=3, min_rgb=0, max_rgb=255):
    b, g, r = cv2.split(img)
    r_mesh = np.linspace(min_rgb, max_rgb, n_segments)
    b_mesh = r_mesh
    g_mesh = r_mesh

    counts_map = dict()
    cnt = 0

    for i in range(1, n_segments):
        for j in range(1, n_segments):
            for k in range(1, n_segments):
                r_min = r_mesh[i - 1]
                r_max = r_mesh[i]
                g_min = g_mesh[j - 1]
                g_max = g_mesh[j]
                b_min = b_mesh[k - 1]
                b_max = b_mesh[k]

                color_voxel = dict()
                color_voxel['group'] = (b_min, b_max, g_min, g_max, r_min, r_max)
                count = np.sum(color_thresh(b, g, r, color_voxel['group']))
                color_voxel['count'] = count

                if count not in counts_map.keys():
                    counts_map[count] = []
                counts_map[count].append(color_voxel)

                cnt += 1

    return counts_map


def find_dominant_colors_gs(frame, n_segments, nms_win):
    # find dominant colors based on grayscale values
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate frequency of pixels in range 0-255
    # histg = cv2.calcHist([gray], [0], None, [256], [0, 256])
    counts, bins = np.histogram(gray, range(257))

    # list_colors = []
    counts_map = dict()
    cnt = 0

    for i in range(n_segments):
        max_c_idx = np.argmax(counts)
        count = counts[max_c_idx]
        # list_colors.append((max_c_idx, count))
        min_counts = max(0, max_c_idx - nms_win)
        max_counts = min(len(counts), max_c_idx + nms_win)
        counts[min_counts:max_counts] = 0

        gray_mask = np.bitwise_and(min_counts <= gray, gray <= max_counts)
        colors = frame[gray_mask]
        # avg_color = np.mean(colors, axis=0)
        min_color = np.min(colors, axis=0)
        max_color = np.max(colors, axis=0)

        color_voxel = dict()
        color_voxel['group'] = (min_color[0], max_color[0], min_color[1], max_color[1], min_color[2], max_color[2])
        color_voxel['count'] = count

        if count not in counts_map.keys():
            counts_map[count] = []
        counts_map[count].append(color_voxel)

        cnt += 1

    return counts_map


def color_histogram(frame, mvizt):
    # sys.path.append('../../')
    # import mviz.viz_tools as mvizt

    n_segments = 3
    n_dominant_colors = 2
    n_mean_shift_iter = 3
    close_ksize = (5, 5)

    # find dominant colors
    counts_map = find_dominant_colors(frame, n_segments)

    # mean-shift on dominant colors (refine color segments)
    counts_map = refine_color_seg_mean_shift(frame, counts_map, n_dominant_colors, n_mean_shift_iter)

    # visualize color segments
    return mvizt.show_color_segments(frame, counts_map, n_dominant_colors, close_ksize)


def color_hist_grayscale(frame, mvizt):
    n_segments = 3
    n_dominant_colors = 3
    n_mean_shift_iter = 3
    close_ksize = (5, 5)
    nms_win = 50
    th_cnt = 100

    # find dominant colors based on grayscale values
    counts_map = find_dominant_colors_gs(frame, n_segments, nms_win)

    # mean-shift on dominant colors (refine color segments)
    counts_map = refine_color_seg_mean_shift(frame, counts_map, n_dominant_colors, n_mean_shift_iter)

    # visualize color segments
    return mvizt.show_color_segments(frame, counts_map, n_dominant_colors, close_ksize)


def refine_color_seg_mean_shift(frame, counts_map, n_dominant_colors, n_mean_shift_iter):
    b, g, r = cv2.split(frame)
    descending_counts = sorted(counts_map.keys(), reverse=True)
    for i in range(n_dominant_colors):
        color_count_key = descending_counts[i]
        color_voxels = counts_map[color_count_key]
        for color_voxel in color_voxels:
            cg = color_voxel['group']
            c_avg = 0.5 * np.array([cg[0] + cg[1], cg[2] + cg[3], cg[4] + cg[5]])
            hb = 0.5 * (cg[1] - cg[0])
            hg = 0.5 * (cg[3] - cg[2])
            hr = 0.5 * (cg[5] - cg[4])

            for i in range(n_mean_shift_iter):
                cg = (c_avg[0] - hb, c_avg[0] + hb, c_avg[1] - hg, c_avg[1] + hg, c_avg[2] - hr, c_avg[2] + hr)
                color_mask = color_thresh(b, g, r, cg)
                c_avg = np.mean(frame[color_mask], axis=0)
                hb = 0.5 * (cg[1] - cg[0])
                hg = 0.5 * (cg[3] - cg[2])
                hr = 0.5 * (cg[5] - cg[4])

            color_voxel['group'] = cg

    return counts_map


def calc_object_center(patch):
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(patch, ddepth, 1, 0)
    dy = cv2.Sobel(patch, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

    th = 100
    mask = mag > th
    w, h = mask.shape

    X = np.repeat(np.arange(0, h).reshape((1, h)), w, axis=0)
    Y = np.repeat(np.arange(0, w).reshape((w, 1)), h, axis=1)
    x_sel = X[mask]
    y_sel = Y[mask]
    obj_x = np.sum(x_sel) / len(x_sel)
    obj_y = np.sum(y_sel) / len(y_sel)

    return np.array([obj_x, obj_y])


def smooth(img, gk_size=(3, 3), mk_size=5):
    # make sure images are grayscale
    gray = img
    if len(gray.shape) >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform gaussian denoising followed by median filtering
    gauss = cv2.GaussianBlur(gray, gk_size, cv2.BORDER_DEFAULT)
    med = cv2.medianBlur(gauss, mk_size)

    return med


def get_intensity_gaussian(image, p0, sigma):
    x0 = p0[0]
    y0 = p0[1]

    sigma3 = sigma * 3

    x_min = int(np.floor(x0 - sigma3))
    x_max = int(np.floor(x0 + sigma3))
    xx = np.arange(x_min, x_max+1)
    n_xx = len(xx)
    xx = xx.reshape((1, n_xx)).squeeze()

    y_min = int(np.floor(y0 - sigma3))
    y_max = int(np.floor(y0 + sigma3))
    yy = np.arange(y_min, y_max+1)
    n_yy = len(yy)
    yy = yy.reshape((n_yy, 1))

    if n_xx > n_yy:
        n_yy += 1
        yyy = np.zeros((3, 1), dtype=int)
        yyy[0] = yy[0] - 1
        yyy[1:3, :] = yy
        yy = yyy
    elif n_xx < n_yy:
        xx = np.concatenate([np.array([xx[0]-1]), xx], axis=0)
        n_xx = len(xx)

    X = np.repeat(xx, n_xx, axis=0)
    X = X.reshape((n_xx ** 2, 1))
    Y = np.repeat(yy, n_yy, axis=0)
    Y = Y.reshape((n_yy ** 2, 1))
    # assert len(X) == len(Y)

    if len(X) != len(Y):
        print('len(X) != len(Y)! -> %d, %d' % (len(X), len(Y)))

    sig2 = sigma ** 2
    if len(image) == 1:
        F = (image / (2 * np.pi * sig2)) * np.exp(((X - x0) ** 2 + (Y - y0) ** 2) / (-2 * sig2))
    else:
        F = (image[X, Y] / (2 * np.pi * sig2)) * np.exp(((X - x0) ** 2 + (Y - y0) ** 2) / (-2 * sig2))
    return X, Y, F


def get_mesh(size):
    xsz = size[0]
    ysz = size[1]
    x = np.arange(ysz)
    y = np.arange(xsz)
    X = np.repeat(x.reshape((1, len(x))), xsz, axis=0)
    Y = np.repeat(y.reshape((len(y), 1)), ysz, axis=1)
    img_mask = np.concatenate((X.reshape(ysz, xsz, 1), Y.reshape(ysz, xsz, 1)), axis=2)

    return X, Y, img_mask


def get_rect_kernel(center, win_size, image_size):
    h, w = image_size
    x, y = center
    hws = int(win_size / 2)
    min_row = max(0, y - hws)
    max_row = min(h, y + hws + 1)
    min_col = max(0, x - hws)
    max_col = min(w, x + hws + 1)

    X = np.arange(min_col, max_col)
    nx = len(X)
    Y = np.arange(min_row, max_row)
    ny = len(Y)
    X = np.repeat([X], ny, axis=0)
    Y = np.repeat(Y.reshape(ny, 1), nx, axis=1)
    Z = np.stack((X, Y), axis=2)

    return Z.reshape((nx * ny, 2))


def get_mesh_kernel(center, win_size, image_size, circle=False):
    h, w = image_size
    if win_size % 2 == 0:
        win_size += 1
    hws = int(win_size / 2)

    X, Y = get_mesh((win_size, win_size))
    X -= hws
    Y -= hws

    XX = X + center[0]
    cx = np.bitwise_and(XX >= 0, XX < w)
    YY = Y + center[1]
    cy = np.bitwise_and(YY >= 0, YY < h)

    if circle:
        r = np.sqrt(0.5 * (X ** 2 + Y ** 2))
        cx = np.bitwise_and(cx, r < hws)
        cy = np.bitwise_and(cy, r < hws)

    cx = cx.flatten()
    cy = cy.flatten()
    cf = np.bitwise_and(cx, cy)

    Z = np.stack((XX, YY), axis=2)
    Z = Z.reshape((win_size ** 2, 2))

    return Z[cf]


def get_neighbors_v(points, win_size, image_size):
    h, w = image_size
    n_points = len(points)
    hws = int(win_size / 2)
    ws2 = win_size ** 2

    # prepare the additive kernel
    # from get_mesh_kernel
    X, Y, _ = get_mesh((win_size, win_size))
    X -= hws
    Y -= hws

    W = np.stack((X, Y), axis=2)
    W = W.reshape((1, ws2, 2))

    Z = np.repeat(W, n_points, axis=0)
    points_reshaped = points.reshape((n_points, 1, 2))
    Z += points_reshaped

    zero_check = Z < 0
    zero_check = np.bitwise_or(zero_check[:, :, 0], zero_check[:, :, 1])
    w_check = Z[:, :, 0] >= w
    h_check = Z[:, :, 1] >= h
    n_mask = np.bitwise_or(h_check, w_check)
    n_mask = np.bitwise_or(zero_check, n_mask)
    mask = ~n_mask

    return Z, mask


def get_mask(point, img_sz, r=30):
    x = point[0]
    y = point[1]
    w = img_sz[1]
    h = img_sz[0]

    xm_min = max(0, int(x - r))
    xm_max = min(w, int(x + r))
    ym_min = max(0, int(y - r))
    ym_max = min(h, int(y + r))

    mask = np.zeros(img_sz, dtype="uint8")
    cv2.rectangle(mask, (xm_min, ym_min), (xm_max, ym_max), 255, -1)

    return mask


def get_img_points_roi(bbox, img_size):
    w, h = img_size[:2]
    x0, x1, y0, y1 = bbox
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, w)
    y1 = min(y1, h)
    xc = np.arange(x0, x1, 1)
    yc = np.arange(y0, y1, 1)
    xcm, ycm = np.meshgrid(xc, yc)
    xyc = np.concatenate([xcm.reshape((-1, 1)), ycm.reshape((-1, 1))], axis=1)
    return xyc


class CPointInfo:
    def __init__(self, point, next_point, next_dir=None, gv_dist=-1):
        self.point = point
        self.next_point = next_point
        self.dir_list = []
        if next_dir is not None:
            self.dir_list.append(next_dir)
        self.gv_dist = -1
        self.max_gv = 0
        self.min_gv = 0
        self.should_stop = False

# deprecated
def segment_contours(img_color, img_gray, cpt_info_list, all_colors=None, seg_win=5, th_gv=0.7, th_dist=10):
    # segment colors along contours
    cnt = 0
    visited_points = []
    for c_pt_info in cpt_info_list:
        if c_pt_info.should_stop:
            continue
        cnt += 1
        # initially, next point is equal to current point
        c_pt = c_pt_info.next_point
        if len(visited_points) <= 0:
            visited_points = np.array([c_pt])
        else:
            if np.any(np.all(c_pt == visited_points, axis=1)):
                continue
            visited_points = np.concatenate((visited_points, [c_pt]), axis=0)
        Z = get_rect_kernel(c_pt, seg_win, img_color.shape[:2])

        # gradient computation: used for init. and stop
        gray_vals = img_gray[Z[:, 1], Z[:, 0]]
        min_gv_idx = np.argmin(gray_vals)
        max_gv_idx = np.argmax(gray_vals)
        min_gv = gray_vals[min_gv_idx]
        max_gv = gray_vals[max_gv_idx]
        gv_dist = max_gv - min_gv

        # calculate next point and direction based on gradient
        if c_pt_info.gv_dist < 0:
            # initial iteration
            c_pt_info.gv_dist = gv_dist
            c_pt_info.min_gv = min_gv
            c_pt_info.max_gv = max_gv
            next_point = Z[max_gv_idx]
            next_dir = next_point - c_pt
        else:
            # next iterations
            # stop cond: if point traveled certain distance or there is a gradient change
            init_gv_dist = c_pt_info.gv_dist
            dist_traveled = point_dist(c_pt, c_pt_info.point)
            c_pt_info.should_stop = abs(gv_dist) > th_gv * init_gv_dist or dist_traveled > th_dist
            # calculate distance between next potential point to determine next point
            next_point_h = c_pt + c_pt_info.dir_list[-1]
            points_dist = Z - next_point_h
            dp2 = np.sum(points_dist ** 2, axis=1)
            next_point = Z[np.argmin(dp2)]
            next_dir = next_point - c_pt

        c_pt_info.next_point = next_point
        c_pt_info.dir_list.append(next_dir)

        colors = img_color[Z[:, 1], Z[:, 0]]
        color_space = np.concatenate((colors, Z), axis=1)
        if all_colors is None:
            all_colors = color_space
        else:
            all_colors = np.concatenate((all_colors, color_space), axis=0)

    # todo: also try color segmentation e.g. watershed
    all_colors = np.unique(all_colors, axis=0)
    return cpt_info_list, all_colors, cnt


def segment_contours_v(img_color, img_gray, img_edge, points, win_size, criteria, img_show):
    c_pts = np.squeeze(points)
    if len(c_pts.shape) != 2:
        c_pts = np.array([c_pts])

    all_points, mask = get_neighbors_v(c_pts, win_size, img_color.shape[:2])
    search_pts = np.unique(all_points[mask], axis=0)

    intensities = img_gray[search_pts[:, 1], search_pts[:, 0]]
    colors = img_color[search_pts[:, 1], search_pts[:, 0]]
    img_single_edge = np.copy(img_edge)
    img_single_edge[c_pts[:, 1], c_pts[:, 0]] = 0
    edge_vals = img_single_edge[search_pts[:, 1], search_pts[:, 0]]

    # detect segments based on: intensities, colors, edges
    # find two modes: dark and light colors (that have caused the gradient here)
    retval, labels, centers = cv2.kmeans(np.float32(colors[:, :3]), 2,
                                         None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels_sqz = np.squeeze(labels)
    cls0_pts = search_pts[labels_sqz == 0]
    cls1_pts = search_pts[labels_sqz == 1]

    # if len(contour) > 500:
    img_show[cls0_pts[:, 1], cls0_pts[:, 0]] = (0, 255, 0)
    img_show[cls1_pts[:, 1], cls1_pts[:, 0]] = (255, 0, 0)
    img_show[c_pts[:, 1], c_pts[:, 0]] = (0, 0, 255)

    return img_show


def get_color_image(image):
    img_show = image
    if len(image.shape) < 3:
        img_show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return img_show


def get_cross_lines(p0, p1):
    # calculate the line between points and the perpendicular line
    # ay + bx + c = 0
    dp = p1 - p0
    pp = (p1 + p0) * 0.5
    dx = dp[0]
    dy = dp[1]
    c = dx * pp[1] - dy * pp[0]
    l = np.array([dx, -dy, -c])
    cp = dy * pp[1] + dx * pp[0]
    lp = np.array([dy, dx, -cp])
    return l, lp


def get_perspective_transform(img, src, dst, size):
    """
    #---------------------
    # This function takes in an image with source and destination image points,
    # generates the transform matrix and inverst transformation matrix,
    # warps the image based on that matrix and returns the warped image with new perspective,
    # along with both the regular and inverse transform matrices.
    #
    """

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def process_detected_contours(image, contours):
    blank = np.zeros(image.shape)
    good_contours = []
    img_sz = image.shape[:2]
    color_thresh = (127, 255, 127, 255, 127, 255)

    for contour in contours:
        c_pts = np.squeeze(contour)
        if len(c_pts.shape) != 2:
            c_pts = np.array([c_pts])
        n_pts = len(c_pts)
        c_pts_img = np.fliplr(c_pts)
        color_vec = image[c_pts_img[:, 0], c_pts_img[:, 1], :]
        m_cv = np.mean(color_vec, axis=0)
        cov_cv = np.cov(color_vec.T, ddof=0)

        aug_cs = np.concatenate((color_vec, c_pts_img), axis=1)
        m_cs = np.mean(aug_cs, axis=0)

        # todo: min/max contour points, corners, closed, open?...
        if n_pts >= 3:
            n_test_pts = 40
            l, lp = get_cross_lines(c_pts[0], c_pts[-1])
            xp, yp = c_pts[int(n_pts * 0.5)]
            lp[2] = -(yp * lp[0] + xp * lp[1])
            if lp[0] == 0:
                # print('horizontal line')
                yp_arr = np.linspace(yp, yp + n_test_pts, n_test_pts)
                xp_arr = np.ones(len(yp_arr)) * (-lp[2] / lp[1])
                pts_test = np.concatenate((xp_arr.reshape((n_test_pts, 1)), yp_arr.reshape((n_test_pts, 1))),
                                          axis=1)
                cv2.line(image, np.int32(pts_test[0, :]), np.int32(pts_test[1, :]), (255, 0, 0), 2)
            elif lp[1] == 0:
                # print('vertical line')
                xp_arr = np.linspace(xp, xp + n_test_pts, n_test_pts)
                yp_arr = np.ones(len(xp_arr)) * (-lp[2] / lp[0])
                pts_test = np.concatenate((xp_arr.reshape((n_test_pts, 1)), yp_arr.reshape((n_test_pts, 1))),
                                          axis=1)
            else:
                # print('slopped line')
                xp_arr = np.linspace(xp, xp + n_test_pts, n_test_pts)
                yp_arr = (-xp_arr * lp[1] - lp[2]) / lp[0]
                pts_test = np.concatenate((xp_arr.reshape((n_test_pts, 1)), yp_arr.reshape((n_test_pts, 1))),
                                          axis=1)

            # cv2.line(img_crop, np.int32(pts_test[0, :]), np.int32(pts_test[-1, :]), (255, 0, 0), 1)
            # cv2.drawMarker(img_crop, (xp, yp), (0, 255, 0))


def calc_bounds(bounds, offset):
    os_start, os_end = offset
    b_start, b_end = bounds
    r0 = max(os_start + b_start, b_start)
    if r0 > b_end:
        r0 = b_end
    r1 = min(os_end + b_end, b_end)
    if r1 < b_start:
        r1 = b_start
    if r0 >= r1:
        r0 = b_start
        r1 = b_end
    return r0, r1


def compute_crop(img_size, crop_size):
    start_row, end_row, start_col, end_col = np.int32(crop_size)
    rows, cols = img_size[:2]
    r0, r1 = calc_bounds((0, rows), (start_row, end_row))
    c0, c1 = calc_bounds((0, cols), (start_col, end_col))
    return np.int32([r0, r1, c0, c1])


def img_crop(img, rect):
    start_row, end_row, start_col, end_col = rect
    img_sz = img.shape
    if len(img_sz) == 3:
        return img[start_row:end_row, start_col:end_col, :]
    elif len(img_sz) == 2:
        return img[start_row:end_row, start_col:end_col]
    return img


def abs_sobel_thresh(img, orient='x', thresh=(20, 100)):
    """
    #---------------------
    # This function applies Sobel x or y, and then
    # takes an absolute value and applies a threshold.
    #
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    #---------------------
    # This function takes in an image and optional Sobel kernel size,
    # as well as thresholds for gradient magnitude. And computes the gradient magnitude,
    # applies a threshold, and creates a binary output image showing where thresholds were met.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    """
    #---------------------
    # This function applies Sobel x and y,
    # then computes the direction of the gradient,
    # and then applies a threshold.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    # and calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # Return the binary image
    return binary_output.astype(np.uint8)


def get_combined_gradients(img_gray, thresh_x, thresh_y, thresh_mag, thresh_dir):
    """
    #---------------------
    # This function isolates lane line pixels, by focusing on pixels
    # that are likely to be part of lane lines.
    # I am using Red Channel, since it detects white pixels very well.
    #
    """
    sobelx = abs_sobel_thresh(img_gray, 'x', thresh_x)
    sobely = abs_sobel_thresh(img_gray, 'y', thresh_y)
    mag_binary = mag_thresh(img_gray, 3, thresh_mag)
    dir_binary = dir_thresh(img_gray, 15, thresh_dir)

    # combine sobelx, sobely, magnitude & direction measurements
    gradient_combined = np.zeros_like(dir_binary).astype(np.uint8)
    gradient_combined[
        ((sobelx > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobelx > 1) & (sobely > 1))] = 255  # | (R > 1)] = 255

    return gradient_combined


def channel_thresh(channel, thresh=(80, 255)):
    """
    #---------------------
    # This function takes in a channel of an image and
    # returns thresholded binary image
    #
    """
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
    return binary


def get_combined_hls(img, th_h, th_l, th_s):
    """
    #---------------------
    # This function takes in an image, converts it to HLS colorspace,
    # extracts individual channels, applies thresholding on them
    #
    """

    # convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    H, L, S = cv2.split(hls)

    h_channel = channel_thresh(H, th_h)
    l_channel = channel_thresh(L, th_l)
    s_channel = channel_thresh(S, th_s)

    # Trying to use Red channel, it works even better than S channel sometimes,
    # but in cases where there is shadow on road and road color is different,
    # S channel works better.
    hls_comb = np.zeros_like(s_channel).astype(np.uint8)
    hls_comb[((s_channel > 1) & (l_channel == 0)) | ((s_channel == 0) & (h_channel > 1) & (l_channel > 1))] = 255

    # return combined hls image
    return hls_comb


def combine_grad_hls(grad, hls):
    """
    #---------------------
    # This function combines gradient and hls images into one.
    # For binary gradient image, if pixel is bright, set that pixel value in resulting image to 255
    # For binary hls image, if pixel is bright, set that pixel value in resulting image to 255
    # Edit: Assign different values to distinguish them
    #
    """
    result = np.zeros_like(hls).astype(np.uint8)
    # result[((grad > 1) | (hls > 1))] = 255
    result[(grad > 1)] = 255    # 100
    result[(hls > 1)] = 255    # 255

    return result


def get_line_points(pt1, pt2, step_size=1):
    x1, y1 = pt1
    x2, y2 = pt2

    dy = y2 - y1
    dx = x2 - x1
    if dx == 0:
        n_pts = np.ceil(abs(y2 - y1) / step_size) + 1
        yl = np.linspace(y1, y2, int(n_pts), dtype=np.float32)
        xl = np.zeros_like(yl)
    else:
        a = dy / dx
        b = y1 - a * x1

        n_pts = np.ceil(abs(x2 - x1) / step_size) + 1
        xl = np.linspace(x1, x2, int(n_pts), dtype=np.float32)
        yl = a * xl + b

    return np.int32(np.concatenate((xl.reshape((-1, 1)), yl.reshape(-1, 1)), axis=1))


def get_line_points_from_segments(pt_list, line_step=1):
    last_pt = None
    line_pts = None
    for pt in pt_list:
        if last_pt is not None:
            l = get_line_points(last_pt, pt, line_step)
            if line_pts is None:
                line_pts = l
            else:
                line_pts = np.concatenate((line_pts, l))
            # img_show = cv2.line(img_show, last_pt, pt, colors[i % 3], 2)
        last_pt = pt

    line_pts = np.unique(line_pts, axis=0)
    # img_show[line_pts[:, 1], line_pts[:, 0]] = colors[i % 3]
    return line_pts


def print_stats(ft_vec, heading_str, prefix=''):
    min_bgr = np.min(ft_vec, axis=0).reshape((-1, 1))
    max_bgr = np.max(ft_vec, axis=0).reshape((-1, 1))
    med_bgr = np.median(ft_vec, axis=0).reshape((-1, 1))
    mean_bgr = np.mean(ft_vec, axis=0).reshape((-1, 1))
    std_bgr = np.std(ft_vec, axis=0).reshape((-1, 1))
    stats = np.concatenate((min_bgr, max_bgr, med_bgr, mean_bgr, std_bgr), axis=1)
    if heading_str is not None:
        print(heading_str)
    for row in stats:
        print(prefix + str(row))


def print_img_stats(all_pts, color_frames, color_labels, gray_frames, gray_labels, kmeans_criteria=None):
    kmeans_labels = []

    if len(all_pts) <= 0:
        print('print_img_stats: error, no image points provided')
        return kmeans_labels

    if len(color_frames) <= 0 or len(gray_frames) <= 0:
        print('print_img_stats: error, must include at least one color frame and one gray frame')
        return kmeans_labels

    assert len(color_frames) == len(color_labels) and len(gray_frames) == len(gray_labels), \
        'print_img_stats: error, image-label mismatch'

    Y = all_pts[:, 1]
    X = all_pts[:, 0]
    gray = gray_frames[0]
    if kmeans_criteria is None:
        kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    gray_pts = gray[Y, X]
    gray_pts = gray_pts.reshape(-1, 1)

    # print (min, max, med, mean, std) for
    print('[min, max, median, mean, std] for:')

    # BGR, Gray, HSV for each group and
    for i, color_img in enumerate(color_frames):
        color_pts = color_img[Y, X]
        ft_space = np.concatenate((color_pts, gray_pts), axis=1)

        # K-Means clustering BGR color-space
        retval, labels, centers = cv2.kmeans(np.float32(ft_space), 2, None,
                                             kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels_sqz = np.squeeze(labels)
        class0 = ft_space[labels_sqz == 0]
        class1 = ft_space[labels_sqz == 1]

        kmeans_labels.append(labels_sqz)

        print_stats(class0[:, :3], f'\t{color_labels[i]}-Class0:', '\t\t')
        print_stats(class1[:, :3], f'\t{color_labels[i]}-Class1:', '\t\t')
        print_stats(class0[:, 3].reshape((-1, 1)), f'\tGray-{color_labels[i]}-Class0/Class1:', '\t\t')
        print_stats(class1[:, 3].reshape((-1, 1)), None, '\t\t')

    # grad_x, grad_y, grad_mag, grad_ang, and more!
    gray_ft_list = []
    for gray_img in gray_frames[1:]:
        gray_ft_list.append(gray_img[Y, X].reshape(-1, 1))
    grad_fts = np.concatenate(gray_ft_list, axis=1)

    heading_str = '\tGradient-' + '/'.join(gray_labels[1:])
    print_stats(grad_fts, heading_str, '\t\t')
    # print stats for binary images (canny)?

    return kmeans_labels


def nothing(x):
    pass


def t_color_seg_contours(frame):

    # preprocessing [optional]
    # color transform: isolate the best response color
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gradients and thresholding
    img_canny = cv2.Canny(img_gray, 120, 240)

    # smart thresholding
    contours, h = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # segment colors along contours
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    img_show = np.copy(frame)
    seg_win = 41

    for contour in contours:
        if len(contour) < 10:
            continue

        # cpt_info_list = []
        # for c_pt in contour:
        #     c_pt = np.squeeze(c_pt)
        #     cpt_info_list.append(cvu.CPointInfo(c_pt, c_pt))

        # all_colors = None
        # for i in range(2):
        #     cpt_info_list, all_colors, cnt = cvu.segment_contours(frame, img_gray, cpt_info_list, all_colors,
        #                                                           seg_win, 0.9, 40)
        #     if cnt <= 0:
        #         break

        img_show = segment_contours_v(frame, img_gray, img_canny, contour, seg_win, criteria, img_show)

        # cv2.imshow('Segmented Image', img_show)
        # cv2.waitKey(1)

    return img_show


def t_img_resize(img, scale):
    img_resized = img_resize(img, scale)
    cv2.imshow("Resized Image", img_resized)
    cv2.waitKey(0)


def t_color_space(img, action):
    img_new = img
    if action == 'gray':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Result", img_new)
    cv2.waitKey(0)


def t_im_filters(img, action):
    img_new = img
    if action == 'gauss':
        img_new = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    cv2.imshow("Result", img_new)
    cv2.waitKey(0)


def t_canny(img):
    canny = cv2.Canny(img, 127, 255)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)


def t_morph(img):
    dilated = cv2.dilate(img, (3, 3), iterations=5)
    eroded = cv2.erode(dilated, (3, 3), iterations=5)


def t_tresh(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresh', thresh)
    cv2.waitKey(0)


def t_contours(img):
    contours, h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(img.shape)
    blank = cv2.drawContours(blank, contours, -1, (0, 0, 255), 2)
    cv2.imshow('Contours', blank)
    cv2.waitKey(0)


def main():
    from glob import glob
    for fn in glob(os.getenv("DATA_PATH") + '/another-square.jpg'):
        img = cv2.imread(fn)
        # squares = find_squares(img)
        # cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
        cv2.imshow('squares', img)
        ch = cv2.waitKey()
        if ch == 27:
            break

    print('Done')


def t_trackbar():
    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)

    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv2.destroyAllWindows()


def t_resize_and_save(path_img, img_sc, grid=10):
    img = cv2.imread(path_img)
    img_sz = img_resize(img, img_sc)

    h, w = img_sz.shape[:2]
    xg = np.arange(0, w, grid)
    yg = np.arange(0, h, grid)
    X, Y = np.meshgrid(xg, yg)
    xy_grid = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)
    xy_points = xy_grid.reshape(xy_grid.shape[0] * xy_grid.shape[1], 2)

    path_root = os.path.split(path_img)[0]
    new_img = os.path.join(path_root, 'image0.png')
    cv2.imwrite(new_img, img_sz)

    img_show = np.copy(img_sz)
    # for x in xg:
    #     img_show = cv2.line(img_show, (x, 0), (x, h), (255, 0, 0), 1)
    # for y in yg:
    #     img_show = cv2.line(img_show, (0, y), (w, y), (0, 0, 255), 1)
    img_show[xy_points[:, 1], xy_points[:, 0]] = (0, 255, 0)

    cv2.imshow('Image', cv2.rotate(img_show, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey()

    cv2.destroyAllWindows()


def t_other():
    img_dir = os.getenv("IMG_PATH", '.')

    img = cv2.imread(os.path.join(img_dir, 'npr.brightspotcdn.webp'))
    img1 = cv2.imread(os.path.join(img_dir, '1665339569_241056_url.jpeg'))
    img2 = cv2.imread(os.path.join(img_dir, 'creative-learning-objects-on-a-wooden-table-GND52K.jpg'))
    img = cv2.imread(os.path.join(img_dir, 'dice-on-a-craps-table-CX6M3W.jpg'))
    # assert img is not None, "file could not be read, check with os.path.exists()"
    #
    # main()
    # cv2.destroyAllWindows()

    img_size = (480, 640)
    win_size = 5
    Z = get_mesh_kernel((230, 321), win_size, img_size)
    Z = get_mesh_kernel((230, 321), win_size, img_size, True)

    test_points = np.array([[0, 1], [2, 3], [4, 5]])
    win_size = 3
    W, mask = get_neighbors_v(test_points, win_size, img_size)
    # print(W[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
        Testing basic IMP tools
    ''')
    parser.add_argument('path', help='path to the image source')
    args = parser.parse_args()

    # t_trackbar()
    t_resize_and_save(args.path, 0.1)

