#!/usr/bin/env python
import argparse
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

sys.path.append('../mviz/')
sys.path.append('../data_loader/')
from data_loader.image_loader import ImageLoader
from cam_calib_models import CalibPinholeRadTan


class CamCalibMono:
    def __init__(self, img_loader: ImageLoader, grid: tuple=(6, 9), scale=1.0):

        self.img_loader = img_loader

        # Defining the dimensions of checkerboard
        self.grid = grid    # chessboard size
        self.scale = scale  # grid square scale (in centimeters)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.grid[0] * self.grid[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.grid[0], 0:self.grid[1]].T.reshape(-1, 2)
        self.objp *= self.scale

        # prev_img_shape = None

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = []

        self.img_size = []
        self.n_images = 0

        self.calib = CalibPinholeRadTan(None)


    def add_image(self, gray):

        ret, corners = cv2.findChessboardCorners(gray, self.grid, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = corners
        if ret:
            self.objpoints.append(self.objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            self.imgpoints.append(corners2)

            self.n_images += 1

        return ret, corners2

    def load_images(self, img_show=False):

        cnt = 0
        while self.img_loader.is_ok():

            frame = self.img_loader.get_next()
            cnt += 1
            if frame is None:
                print('detected null frame')
                continue

            img = frame.frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = self.add_image(gray)

            # Draw and display the corners
            if img_show and ret:
                img = cv2.drawChessboardCorners(img, self.grid, corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)

            if len(self.img_size) <= 0:
                self.img_size = gray.shape[::-1]
                # print(self.img_size)

        print('-- Successfully added ' + str(self.n_images) + ' images from total ' + str(cnt) + ' images')

    def calibrate(self, K=None, D=None):
        # Always load images before calling this
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, K, D)

        # update intrinsics
        self.calib.update_distortion(dist.squeeze())
        self.calib.update_intrinsics(mtx)

        # set the last image's pose
        rot_obj = scipyR.from_rotvec(rotvec=rvecs[-1].squeeze())
        R_cw = rot_obj.as_matrix()
        t_cw = tvecs[-1].squeeze()
        self.calib.update_pose(R_cw, t_cw)

        self.calib.print()

        print('-- Calibrated Successfully')

        return ret, mtx, dist, rvecs, tvecs

    def recalib(self, cam_obj, new_img, ws=1.0):

        if cam_obj is None:
            return False

        # include old images
        self.load_images()

        if self.n_images < 9:
            return False

        # add the new image
        corners = np.array([])
        if len(new_img) > 0:
            new_gray = new_img
            if len(new_gray.shape) > 2:
                new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            ret, corners = self.add_image(new_gray)
            if not ret:
                return False

        # calibrate
        self.objpoints = ws * np.float32(self.objpoints)
        ret, mtx, dist, rvecs, tvecs = self.calibrate()

        # the last estimate is what we want
        robj = scipyR.from_rotvec(rvecs[-1].reshape((1, 3)).squeeze())
        R_cw = robj.as_matrix()
        t_cw = tvecs[-1].reshape((1, 3)).squeeze()

        cam_obj.update_pose(R_cw, t_cw)
        cam_obj.update_intrinsics(mtx)
        cam_obj.D = dist.squeeze()
        cam_obj.img_size = self.img_size
        cam_obj.calibrated = True

        cam_obj.update_homography()

        if corners is not None and len(corners) > 0:
            pts_img = corners.squeeze()
            cam_obj.ax_x_img = [pts_img[0], pts_img[1]]
            cam_obj.ax_y_img = [pts_img[0], pts_img[self.grid[0]]]

        return True

        # pts_w = ws * np.float32(objp).squeeze()
        # calc_reprojection_error(cam_obj, pts_w, pts_img)
        # test_homography(cam_obj, pts_w, pts_img)

    @staticmethod
    def refine_intrinsics(cam_obj, img, show_img=False):

        K = cam_obj.K
        D = cam_obj.D
        img_size = cam_obj.img_size
        # Refining the camera matrix using parameters obtained by calibration
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)

        # Method 1 to undistort the image
        dst = cv2.undistort(img, K, D, None, K_new)

        # Method 2 to undistort the image
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K_new, img_size, 5)

        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        if show_img:
            # Displaying the undistorted image
            cv2.imshow("undistorted image", dst)
            cv2.waitKey(0)


def calc_reprojection_error(cam_model, pts_w, pts_img):
    if cam_model is None:
        return -1

    # TEST 1: Reprojection Error
    T_cw = cam_model.T_cw
    robj = scipyR.from_matrix(cam_model.R_cw)
    rvec = robj.as_rotvec()
    tvec = cam_model.t_cw
    K = cam_model.K
    D = cam_model.D

    # use opencv project method
    res, other = cv2.projectPoints(pts_w, rvec, tvec, K, D, pts_img)

    # project the hard way!
    pts_w_h = np.concatenate([pts_w, np.ones((len(pts_w), 1))], axis=1).T

    Pc = T_cw[0:3, :] @ pts_w_h
    depth_pc = np.copy(Pc[2, :])
    Pc /= depth_pc

    # xy = K @ Pc
    # xy = xy[0:2, :].T

    Pc_p = cv2.undistortPoints(pts_img, K, D).squeeze()

    err = abs(res.squeeze() - pts_img)
    sum_err = np.sum(np.sqrt(np.sum(err * err, axis=1)))
    err1 = abs(Pc[0:2, :].T - Pc_p)
    sum_err1 = np.sum(np.sqrt(np.sum(err1 * err1, axis=1)))
    print('-- Sum of projection error (OpenCV, on Image): %.4f (pixels)' % sum_err)
    print('-- Sum of reprojection error (Camera Frame): %.4f (cm)' % sum_err1)

    return sum_err1


def t_homography(cam_model, pts_w, pts_img):
    if cam_model is None:
        return

    T_cw = cam_model.T_cw
    K = cam_model.K
    D = cam_model.D

    # TEST 2: Homographies
    H = np.eye(3)
    H[:, 0:2] = T_cw[0:3, 0:2]
    H[:, 2] = T_cw[0:3, 3]
    Hp = K @ H

    H_1 = np.linalg.inv(H)
    Hp_1 = np.linalg.inv(Hp)

    pts_img_norm = cv2.undistortPoints(pts_img, K, D).squeeze()
    upts_homo = np.concatenate([pts_img_norm, np.ones((len(pts_img_norm), 1))], axis=1).T

    XY_orig = np.concatenate([pts_w[:, 0:2], np.ones((len(pts_w), 1))], axis=1)

    # Projection
    xy_p = Hp @ XY_orig.T
    dd = np.copy(xy_p[2, :])
    xy_pp = xy_p / dd  # these are image points if you distort them with D

    # Unprojection
    # remember the points from cv.undistortPoints are normalized by K
    # (so don't use K_1 in the following equation)
    pc = H_1 @ upts_homo

    s_pc = np.copy(pc[2, :])

    XY_homo = pc / s_pc
    XY_homo = XY_homo.T

    err = abs(XY_homo[:, 0:2] - pts_w[:, 0:2])


def show_img_axis(img, cam_obj):
    img = cv2.line(img, np.int32(cam_obj.ax_x_img[0]), np.int32(cam_obj.ax_x_img[1]), (0, 0, 255))
    img = cv2.line(img, np.int32(cam_obj.ax_y_img[0]), np.int32(cam_obj.ax_y_img[1]), (0, 255, 0))
    img = cv2.circle(img, np.int32(cam_obj.ax_x_img[0]), 3, (255, 0, 0))

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)


# this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(cam_obj.unproject_homo([x, y]))
        print([x, y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Camera Calibration, estimate camera parameters from images in a folder
    ''')
    parser.add_argument('img_path', help='path to the images folder')
    parser.add_argument('--img_mode', help='image load mode (default: image_folder)',
                        default='image_folder')
    parser.add_argument('--grid_size', nargs='+', help='grid size (default: (6, 9))', default=[])
    parser.add_argument('--grid_scale', help='grid square scale (default: 1.0 unit)', default=1.0)
    args = parser.parse_args()

    img_path = args.img_path
    grid_size = args.grid_size
    if len(grid_size) > 0:
        grid_size = np.int32(grid_size)
    else:
        grid_size = (6, 9)
    grid_scale = float(args.grid_scale)

    image_loader = ImageLoader(img_path, args.img_mode)

    camCalib = CamCalibMono(image_loader, grid_size, grid_scale)
    camCalib.load_images(img_show=False)
    # camCalib.add_image(new_img)
    camCalib.calibrate()
    calc_reprojection_error(camCalib.calib, camCalib.objpoints[-1].squeeze(), camCalib.imgpoints[-1].squeeze())



