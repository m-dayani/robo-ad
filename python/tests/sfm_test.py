"""
    Test Structure From Motion:
        Known Params:   Camera poses, camera calibration, image observation matches (e.g. key points),
                        and possibly, pixel depth
        Estimate:       World objects (e.g. world point cloud)

    NOTE:
        There are lots of similar methods elsewhere (e.g. camera.stereo.stereo_mapping.py)
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
sys.path.append('../data_loader')
sys.path.append('../mviz')
sys.path.append('../cv_mine/')
sys.path.append('../cv_mine/obj_det_tr')
from data_loader.image_loader import ImageLoader, ImageLoaderEuRoC
from data_loader.pose_loader import PoseLoader
from calib.cam_calib_models import CalibPinholeRadTan
from tools.utils import load_settings
from mviz.viz_scene import draw_camera_and_wpts
from cv_mine.obj_det_tr.detection import ObjectDetection
from cv_mine.obj_det_tr.detection_dl_tflite import ObjectDetectionTflite
from cv_mine.cv_utils import get_img_points_roi, get_color_image, img_resize
from camera.stereo_cam import MultiCam
from my_geo.world_objects import WorldObject


left_clicks0 = list()
left_clicks1 = list()
img = None


def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global left_clicks0
        global left_clicks1
        if params == 0:
            left_clicks0.append([x, y])
        elif params == 1:
            left_clicks1.append([x, y])


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


class KptDetMatch:
    def __init__(self, n_kpts=400):
        self.detector = cv2.ORB_create(n_kpts)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_match(self, img0, img1):
        # find the key points and descriptors with ORB
        kp1, des1 = self.detector.detectAndCompute(img0, None)
        kp2, des2 = self.detector.detectAndCompute(img1, None)

        # Match descriptors.
        matches = self.matcher.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, des1, kp2, des2, matches

    def match_stereo_planar(self, wobj0, img0, img1):
        # map the center point of wobj0 to img1
        pt0 = wobj0.get_center()
        pt1_p = self.E_c1c0 @ self.get_point_h(pt0)

        # mask an area around the mapped point
        mask1 = np.zeros(img1.shape[:2])
        mask1 = cv2.circle(mask1, (pt1_p[0], pt1_p[1]), 20, 255, -1)
        img2 = cv2.bitwise_and(mask1, img1)

        # detect features and descriptors in this area
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        # detect a descriptor at the initial point
        des0 = self.orb.compute(img0, pt0)

        # match features
        matches = self.bf_matcher.match([des0], des2)

        # traverse over matches and choose the point with: min(d_desc * d_point)
        min_dist = 1e9
        min_pt = pt1_p
        for match in matches:
            pt1 = kp2[match.trainIdx]
            d_point = np.linalg.norm(np.array((pt1.x, pt1.y)) - np.array((pt1_p[0], pt1_p[1])))
            d_desc = match.distance
            if d_desc * d_point < min_dist:
                min_dist = d_desc * d_point
                min_pt = pt1
        return min_pt

    def match_stereo_epipolar(self, wobj0, img0, img1):
        # wobj0 must be rectified before sending to this
        # kpt_d = cv2.KeyPoint()
        # kpt_d.pt = wobj0.center
        kpt_d = wobj0.kpt

        # search around this point for matches
        mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
        mask1 = cv2.circle(mask1, np.int32((kpt_d.pt[0], kpt_d.pt[1])), 50, 255, -1)

        pt2, des2 = self.orb.detectAndCompute(img1, mask1)

        if len(pt2) <= 0:
            return kpt_d.pt

        # detect a descriptor at the initial point
        kpt0, des0 = self.orb.compute(img0, np.array([kpt_d]))

        # match features
        matches = self.bf_matcher.match(des0, des2)

        # traverse over matches and choose the point with: min(d_desc * d_point)
        min_dist = 1e9
        min_pt = kpt_d.pt
        for match in matches:
            pt1 = pt2[match.trainIdx].pt
            # d_point = np.linalg.norm(np.array((pt1[0], pt1[1])) - np.array((pc1_s1[0], pc1_s1[1])))
            d_desc = match.distance
            if d_desc < min_dist:
                min_dist = d_desc
                min_pt = pt1
        return min_pt

    def match_stereo_matching(self, wobj0, img0, img1):
        # You need the keypoint (not just point) for this matching,
        # but you only have point in most cases!
        # one solution is to detectAndCompute in vicinity of both points in images
        # and match those (you can find camera 2D image transformations experimentally ->
        # then use it for matching)
        # NOTE: All images are distorted, and coord in wobj0 must be assigned correctly

        pt0 = wobj0.center
        pt0_ud = wobj0.center_ud
        pt1_ud = (self.H01 @ self.get_point_h(pt0_ud)).reshape((1, 3)).squeeze()[:2]
        pt1 = self.distort_point(pt1_ud, 1)

        mask0 = np.zeros(img0.shape[:2], dtype=np.uint8)
        mask0 = cv2.circle(mask0, np.int32(pt0[:2]), 50, 255, -1)
        mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
        mask1 = cv2.circle(mask1, np.int32(pt1[:2]), 50, 255, -1)

        kpt0, des0 = self.orb.detectAndCompute(img0, mask0)
        kpt1, des1 = self.orb.detectAndCompute(img1, mask1)

        if kpt0 is None or kpt1 is None or len(kpt0) <= 0 or len(kpt1) <= 0:
            return pt0, pt1

        matches = self.bf_matcher.match(des0, des1)

        if matches is None:
            return pt0, pt1

        d_min = 1e9
        p0 = pt0
        p1 = pt1
        for match in matches:
            kp0 = kpt0[match.queryIdx]
            kp1 = kpt1[match.trainIdx]
            d_desc = match.distance
            d0 = np.linalg.norm(kp0.pt - pt0)
            d1 = np.linalg.norm(kp1.pt - pt1)
            d_tot = d_desc * d0 * d1
            if d_min > d_tot:
                p0 = kp0.pt
                p1 = kp1.pt
                d_min = d_tot

        return p0, p1


class StereoMapping:
    def __init__(self, settings):
        self.T_bc0 = np.array(settings['T_bc0']).reshape((4, 4))
        self.T_c0b = np.linalg.inv(self.T_bc0)
        self.H_c0b = self.get_homography(self.T_bc0)
        intrinsics0 = settings['intrinsics0']
        self.K0 = np.array([[intrinsics0[0], 0.0, intrinsics0[2]],
                            [0.0, intrinsics0[1], intrinsics0[3]],
                            [0.0, 0.0, 1.0]])
        self.D0 = np.array(settings['dist_coeffs0'])
        self.P0 = self.K0 @ self.T_c0b[0:3, :]

        self.T_bc1 = np.array(settings['T_bc1']).reshape((4, 4))
        self.T_c1b = np.linalg.inv(self.T_bc1)
        self.H_c1b = self.get_homography(self.T_c1b)
        intrinsics1 = settings['intrinsics1']
        self.K1 = np.array([[intrinsics1[0], 0.0, intrinsics1[2]],
                            [0.0, intrinsics1[1], intrinsics1[3]],
                            [0.0, 0.0, 1.0]])
        self.D1 = np.array(settings['dist_coeffs1'])
        self.P1 = self.K1 @ self.T_c1b[0:3, :]
        # self.E_c1c0 = self.K1 @ self.H_c1b @ np.linalg.inv(self.H_c0b) @ np.linalg.inv(self.K0)
        self.H01 = np.array(settings['H_stereo']).reshape((3, 3))

        self.orb = cv2.ORB_create(100)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def calc_stereo_line(self, kpt):
        pc0 = np.linalg.inv(self.K0) @ self.get_point_h(kpt)
        # these define your ray in cam0
        lc0_s = self.T_bc0[:3, :3] @ pc0
        lc0_p = self.T_bc0[:3, 3].reshape((3, 1))

        # corresponding line in cam1
        lc1_s = self.K1 @ self.T_c1b[:3, :3] @ lc0_s
        lc1_p = self.K1 @ (self.T_c1b[:3, :3] @ lc0_p + self.T_c1b[:3, 3].reshape((3, 1)))

        pc1_s1 = (lc1_s + lc1_p).reshape((1, 3)).squeeze()
        pc1_s1 /= pc1_s1[2]
        pc1_s1 = pc1_s1.reshape((1, 3)).squeeze()

        return lc1_s, lc1_p, pc1_s1

    def compute_homography_points(self, corners0, corners1):
        # Note: even though this homography is computed for image plains,
        # it only works for planar objects inside images (not objects at arbitrary depth!)
        self.H01 = cv2.getPerspectiveTransform(np.array(corners0[:4], dtype=np.float32), np.array(corners1[:4], dtype=np.float32))
        return self.H01

    def project_homography(self, H, X, cam=0):
        if cam == 0:
            K = self.K0
        else:
            K = self.K1
        x = K @ H @ self.get_point_h(X)
        x = x.reshape((1, 3)).squeeze()
        x /= x[2]

        return x

    @staticmethod
    def get_homography(T_cb):
        H_cb = np.eye(3)
        H_cb[:, :2] = T_cb[:3, :2]
        H_cb[:, 2] = T_cb[:3, 3]
        return H_cb

    @staticmethod
    def get_stereo_line(img_size, lc1_s, lc1_p):

        h, w = img_size
        s_vec = np.linspace(0, 100, 1000)
        list_pc1 = []
        for s in s_vec:
            pc1 = lc1_s * s + lc1_p
            pc1 /= pc1[2]
            xc1 = pc1[0].squeeze()
            yc1 = pc1[1].squeeze()
            if 0 <= xc1 < w and 0 <= yc1 < h:
                list_pc1.append((xc1, yc1))
        return list_pc1

    @staticmethod
    def compute_homography(img0, img1, grid):
        # this method computes a homography between stereo frames
        # used for feature matching,
        # use this method with images of calibration pattern for best results
        # detect and match calibration pattern,
        # use cv's detectChessboard for this pattern
        # kpt0, des0 = self.orb.detectAndCompute(img0, None)
        # kpt1, des1 = self.orb.detectAndCompute(img1, None)
        # matches = self.bf_matcher.match(des0, des1)

        if len(img0.shape) > 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        if len(img1.shape) > 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        ret0, corners0 = cv2.findChessboardCorners(img0, grid)
        ret1, corners1 = cv2.findChessboardCorners(img1, grid)

        if not ret0 or not ret1:
            return np.eye(3)

        # refine
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners02 = cv2.cornerSubPix(img0, corners0, (11, 11), (-1, -1), criteria)
        corners12 = cv2.cornerSubPix(img1, corners1, (11, 11), (-1, -1), criteria)

        # compute homography
        H01 = cv2.getPerspectiveTransform(corners02, corners12)
        return H01

    @staticmethod
    def calc_rot_mat_from_h(H):
        R_cw = np.eye(3)
        R_cw[:, :2] = H[:, :2]
        R_cw[:, 2] = np.cross(H[:, 0], H[:, 1])
        # print("det(R_cw): " + str(np.linalg.det(R_cw)))
        W, U, Vt = cv2.SVDecomp(R_cw)
        R_cw = U @ Vt
        det = np.linalg.det(R_cw)
        # print("U: " + str(U))
        # print("Vt: " + str(Vt))
        if det < 0:
            Vt[2, :] *= -1
            R_cw = U @ Vt
        return R_cw

    @staticmethod
    def get_trans_from_rt(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


class Triangulation:
    @staticmethod
    def calculate_XYZ(u, v, Kinv, Rinv, tvec):

        # Solve: From Image Pixels, find World Points
        scalingfactor = 1.0
        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = scalingfactor * uv_1
        xyz_c = Kinv.dot(suv_1)
        xyz_c = xyz_c - tvec
        XYZ = Rinv.dot(xyz_c)

        return XYZ

    @staticmethod
    def triangulate_orbslam(x_c1, x_c2, Tc1w, Tc2w):
        # This produces wrong results??
        A = np.zeros((4, 4))
        A[0, :] = x_c1[0] * Tc1w[2, :] - Tc1w[0, :]
        A[1, :] = x_c1[1] * Tc1w[2, :] - Tc1w[1, :]
        A[2, :] = x_c2[0] * Tc2w[2, :] - Tc2w[0, :]
        A[3, :] = x_c2[1] * Tc2w[2, :] - Tc2w[1, :]

        U, s, Vh = np.linalg.svd(A, full_matrices=False)

        x3d_h = Vh[:, 3].reshape((1, 4)).squeeze()
        if abs(x3d_h[3]) < 1e-3:
            return np.zeros((1, 3)).squeeze()

        x3d_h /= x3d_h[3]
        return x3d_h[0:3]

    @staticmethod
    def triangulate_tb(P1, P2, xc1, xc2):

        A = [xc1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - xc1[0] * P1[2, :],
             xc2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - xc2[0] * P2[2, :]]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        # print('Triangulated point: ')
        # print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    @staticmethod
    def triangulate_cv(P1, P2, xc1, xc2):
        Pt3D_h = cv2.triangulatePoints(P1, P2, xc1, xc2)
        scale = Pt3D_h[3, 0]
        return Pt3D_h.reshape((1, 4)).squeeze()[0:3] / scale, scale

    @staticmethod
    def reconstruct_kpt(cam0, cam1, kpt0, kpt1, matches):

        # init. camera calib params
        K0 = cam0.K
        K1 = cam1.K
        P0 = cam0.get_camera_matrix()
        P1 = cam1.get_camera_matrix()

        map_points = []
        cnt_good = 0

        for match in matches:
            # prepare matches
            idx0 = match.queryIdx
            idx1 = match.trainIdx
            pt0 = kpt0[idx0].pt
            pt1 = kpt1[idx1].pt
            pt0_ud = cv2.undistortPoints(pt0, K0, cam0.D, None, K0).squeeze()
            pt1_ud = cv2.undistortPoints(pt1, K1, cam1.D, None, K1).squeeze()

            # triangulate
            Xw, s = Triangulation.triangulate_cv(P0, P1, pt0_ud, pt1_ud)

            if np.linalg.norm(Xw) > 1e-2:   # and s > 0:
                # calculate the reprojection error
                Xww = Xw.reshape(1, 3)
                xc0, sc0 = cam0.project(Xww)
                xc1, sc1 = cam1.project(Xww)
                e = np.linalg.norm(xc0.squeeze() - pt0_ud) + np.linalg.norm(xc1.squeeze() - pt1_ud)
                if e < 4 and not (sc0 < 0 or sc1 < 0):
                    cnt_good += 1
                    # print('good point')
                    # O_s0 = Tc1b[0:3, 3]
                    # O_s1 = Tc2b[0:3, 3]
                    # # mviz.draw_line_3d(ax, O_s0, O_s0 + xc1)
                    # mviz.draw_line_3d(ax, O_s1, O_s1 + xc2)
                    # plt.pause(0.001)
                    map_points.append(Xw)

        print(str(cnt_good) + ' good triangulation from ' + str(len(matches)) + ' total matches')
        return map_points


def print_homo(H):
    rows, cols = H.shape
    res = '['
    for i in range(rows):
        for j in range(cols):
            sep = ','
            if i == rows - 1 and j == cols - 1:
                sep = ']'
            res += f'%.4f{sep} ' % H[i, j]

    return res


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def t_depth_estimation(stereo_mapping, img0, img1):
    pt0, des0 = stereo_mapping.orb.detectAndCompute(img0, None)

    kpt = pt0[0].pt
    kpt_ud = cv2.undistortPoints(np.array([kpt]), stereo_mapping.K0, stereo_mapping.D0, None,
                                 stereo_mapping.K0).squeeze()
    pc0 = np.linalg.inv(stereo_mapping.K0) @ stereo_mapping.get_point_h(kpt)

    wobj0 = WorldObject()
    wobj0.center = kpt
    wobj0.center_ud = kpt_ud
    wobj0.kpt = pt0[0]
    # You need the keypoint (not just point) for this matching,
    # but you only have point in most cases!
    # one solution is to detectAndCompute in vicinity of both points in images
    # and match those (you can find camera 2D image transformations experimentally -> then use it for matching)
    pt_match = stereo_mapping.match_stereo_epipolar(wobj0, img0, img1)
    pt_match_ud = cv2.undistortPoints(np.array([kpt]), stereo_mapping.K1, stereo_mapping.D1, None,
                                      stereo_mapping.K1).squeeze()

    Xb = stereo_mapping.triangulate_cv(kpt_ud, pt_match_ud)

    img0_show = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img0_show[:, :, 2] = img1
    # text = f'(%.2f, %.2f, %.2f)' % (Xb[0], Xb[1], Xb[2])
    # img0_show = cv2.drawMarker(img0_show, np.int32(kpt), (255, 0, 0), cv2.MARKER_CROSS, 8, 2)
    # img0_show = cv2.putText(img0_show, text, np.int32(kpt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Cam-Mixed", img0_show)
    cv2.waitKey(1)


def t_apriltag_mapping(stereo_mapping, obj_detector, img0):

    img0_ud = stereo_mapping.undistort_image(img0)
    wobj, img2 = obj_detector.detect_apriltag(img0_ud, True)

    if len(wobj) > 0:
        print('Detected Apriltags: ' + str(len(wobj)))
        # you can compute the homography between the corners and known dimensions
        scale = obj_detector.april_scale
        hscale = scale * 0.5
        # w_pts = np.array([[0.0, 0.0], [0.0, scale], [scale, 0.0], [scale, scale]], dtype=np.float32)
        # H_wc = cv2.getPerspectiveTransform(np.float32(wobj[0].apriltag.corners), w_pts)
        # H_wc = np.linalg.inv(H_cw)
        # convert other points to world using this transform
        pt_c = stereo_mapping.get_point_h(wobj[0].apriltag.corners[2]).reshape((3, 1))
        HH = wobj[0].apriltag.homography # H_cw
        pt_w = np.linalg.inv(HH) @ pt_c
        pt_w = pt_w.reshape((1, 3)).squeeze()
        pt_w = pt_w[:2] / pt_w[2] * hscale

        # the above solution is also wrong for the same reason (Z = 0)
        # you should estimate camera pose from homography
        objectPoints = np.array([[-hscale, -hscale], [hscale, -hscale], [hscale, hscale], [-hscale, hscale]], dtype=np.float32)
        imagePoints = np.array(wobj[0].apriltag.corners, dtype=np.float32)
        # ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, stereo_mapping.K0, None)

        # normalize image points
        norm_image_points = stereo_mapping.normalize_points(imagePoints)

        # A and HH are identical
        # don't normalize the scale, since we have metric distances, the scales are correct
        A = np.linalg.inv(stereo_mapping.K0) @ HH
        HH, _ = cv2.findHomography(objectPoints, norm_image_points)

        # These two are identical -> not always!, but t_cw_a is readily computed
        t_cw = HH[:, 2]
        t_cw_a = A[:, 2]    # / A[2, 2]
        R_cw = StereoMapping.calc_rot_mat_from_h(HH)
        R_cw_a = StereoMapping.calc_rot_mat_from_h(A)

        # These transforms must map the points correctly -> OpenCV's result is more stable
        # This happens when the signs of the first two columns of R_cw mismatch -> why??
        H_cw_a = StereoMapping.get_homography(StereoMapping.get_trans_from_rt(R_cw_a, t_cw_a))
        H_cw = StereoMapping.get_homography(StereoMapping.get_trans_from_rt(R_cw, t_cw))
        pc0_a = stereo_mapping.project_homography(H_cw_a, objectPoints[0])
        r0 = np.linalg.norm(imagePoints[0] - pc0_a[:2])
        pc0 = stereo_mapping.project_homography(H_cw, objectPoints[0])
        r1 = np.linalg.norm(imagePoints[0] - pc0[:2])

        t_wc = -(R_cw.transpose() @ t_cw.reshape((3, 1))).reshape(1, 3).squeeze()
        distance = np.linalg.norm(t_cw_a / scale)
        img2 = cv2.putText(img2, str(distance), np.int32(wobj[0].apriltag.center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        print("t_cw_a: " + str(t_cw_a))
        print("t_wc: " + str(t_wc))
        print("Distance from camera to Apriltag's center: %.2f (m)" % distance)
        print("errors (t_cw, r0, r1): (%.6f, %.2f, %.2f)" % (np.linalg.norm(t_cw - t_cw_a), r0, r1))

    return img2


def t_chessboard_mapping(stereo_mapping, img0, img1):
    # unlike the apriltag, the results of this change realisticly
    # but with a small offset (about 5 cm)
    CHECKERBOARD = (6, 9)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, criteria)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, criteria)

    # img2 = np.copy(img0)

    if ret0 and ret1:
        corners0_ud = stereo_mapping.undistort_point(corners0)
        corners1_ud = stereo_mapping.undistort_point(corners1, cam=1)

        for pt0, pt1 in zip(corners0_ud, corners1_ud):
            Xb = stereo_mapping.triangulate_cv(pt0, pt1)

        print('pt0: ' + str(pt0))
        print('pt1: ' + str(pt1))
        print('Xb: ' + str(Xb))

    return cv2.drawChessboardCorners(img0, CHECKERBOARD, corners0, ret0)


def t_matching(stereo_mapping, img0, img1, points_rectified=False):
    global left_clicks0
    global left_clicks1

    marker_sz = 1
    marker_th = 0.5

    if len(left_clicks0) >= 2:

        mapped_pts_matched = []
        mapped_pts_selected = []
        img_show = get_color_image(img0)
        img_show1 = get_color_image(img1)

        idx = 0
        for pt_d in left_clicks0:

            np_pt_d = np.array(pt_d, dtype=np.float32)
            if points_rectified:
                pt_ud = np_pt_d
                pt_d = stereo_mapping.distort_point(pt_ud)
            else:
                pt_ud = stereo_mapping.undistort_point(np_pt_d)
            cv2.drawMarker(img_show, np.int32(pt_d), (255, 0, 0), cv2.MARKER_CROSS, marker_sz, marker_th)
            cv2.putText(img_show, str(idx), np.int32(pt_d), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.drawMarker(img_show, np.int32(pt_ud), (0, 0, 255), cv2.MARKER_CROSS, marker_sz, marker_th)

            wobj0 = WorldObject()
            wobj0.center = np_pt_d
            wobj0.center_ud = pt_ud

            pt_match0, pt_match1 = stereo_mapping.match_stereo_matching(wobj0, img0, img1)

            cv2.drawMarker(img_show, np.int32(pt_match0), (255, 100, 0), cv2.MARKER_CROSS, marker_sz, marker_th)
            cv2.drawMarker(img_show1, np.int32(pt_match1), (0, 0, 255), cv2.MARKER_CROSS, marker_sz, marker_th)
            cv2.putText(img_show1, str(idx), np.int32(pt_match1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # compute depth using each pair and compare
            pt_match0_ud = stereo_mapping.undistort_point(np.array(pt_match0, dtype=np.float32))
            pt_match1_ud = stereo_mapping.undistort_point(np.array(pt_match1, dtype=np.float32), 1)
            Xb_match = stereo_mapping.triangulate_cv(pt_match0_ud, pt_match1_ud)
            mapped_pts_matched.append(Xb_match)
            print('----------------------------------------------------')
            print('Iteration: #' + str(idx))
            print('Triangulation between matched rectified points:')
            print('Point0: ' + str(pt_match0_ud))
            print('Point1: ' + str(pt_match1_ud))
            print('Xb: ' + str(Xb_match))

            np_pt1_d = np.array(left_clicks1[idx], dtype=np.float32)
            if points_rectified:
                pt1_ud = np_pt1_d
            else:
                pt1_ud = stereo_mapping.undistort_point(np_pt1_d, 1)
            Xb_select = stereo_mapping.triangulate_cv(pt_ud, pt1_ud)
            mapped_pts_selected.append(Xb_select)
            print('----------------------------------------------------')
            print('Triangulation between selected rectified points:')
            print('Point0: ' + str(pt_ud))
            print('Point1: ' + str(pt1_ud))
            print('Xb: ' + str(Xb_select))

            idx += 1

        for i in range(len(mapped_pts_matched)):
            next_i = int((i + 1) % len(mapped_pts_matched))
            Xb0_matched = mapped_pts_matched[i]
            Xb1_matched = mapped_pts_matched[next_i]
            Xb0_selected = mapped_pts_selected[i]
            Xb1_selected = mapped_pts_selected[next_i]

            print('Distance between matched points (%d, %d): %.2f (cm)' %
                  (i, next_i, np.linalg.norm(Xb0_matched - Xb1_matched) * 100))
            print('Distance between selected points (%d, %d): %.2f (cm)' %
                  (i, next_i, np.linalg.norm(Xb0_selected - Xb1_selected) * 100))

        cv2.imshow("Clicks0", img_show)
        cv2.imshow("Clicks1", img_show1)
        cv2.waitKey()

        left_clicks0 = list()
        left_clicks1 = list()


def t_compute_homography(stereo_mapping, img0, img1):
    global left_clicks0
    global left_clicks1

    img0_ud = stereo_mapping.undistort_image(img0)
    img1_ud = stereo_mapping.undistort_image(img1, 1)

    cv2.imshow("image0", img0_ud)
    cv2.imshow("image1", img1_ud)
    cv2.waitKey()

    if 4 <= len(left_clicks1) == len(left_clicks0):
        stereo_mapping.compute_homography_points(left_clicks0, left_clicks1)
        print(stereo_mapping.H01)

        img1_warped_ud = cv2.warpPerspective(img0_ud, stereo_mapping.H01, (img1.shape[1], img1.shape[0]))

        img_show_ud = cv2.merge((img1_ud, img1_warped_ud, img1_warped_ud))

        cv2.imshow("Warped Homography", img_show_ud)
        cv2.waitKey()


def t_constant_homography(stereo_mapping, img0, img1):

    # NOTE: Once you estimate a homography between two planes (camera attached rigidly to car and the road)
    # you can use the transform for other roads (because the relation between planes won't change)
    # For Apriltags you might be able to use PnP or DLT methods for pose estimation
    # For signs and varing depth objects use the general camera model

    img0_ud = stereo_mapping.undistort_image(img0)
    img1_ud = stereo_mapping.undistort_image(img1, 1)

    img1_warped_ud = cv2.warpPerspective(img0_ud, stereo_mapping.H01, (img1.shape[1], img1.shape[0]))

    img_show_ud = cv2.merge((img1_ud, img1_warped_ud, img1_warped_ud))

    cv2.imshow("Warped Homography", img_show_ud)
    cv2.waitKey(1)


def t_perspective_correction(settings, img_path, img1_path='', pattern_size=(6, 9), cam=0):

    if len(img_path) <= 0 or not os.path.exists(img_path):
        print('Could not find image: ' + img_path)
        return

    cam_calib = CalibPinholeRadTan(settings)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    gray = img
    img_warp = img
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria_calib = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria_corners = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, criteria_calib)

    px_scale = 30
    px_start = 130
    px_end = 110
    warp_size = (340, 360)

    if ret:
        if len(img1_path) <= 0 or not os.path.exists(img1_path):
            # Find trans between world points and image points
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 2), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * px_scale
            # objp = objp[:, :2]
            objp[:, 0] += px_start
            objp[:, 1] += px_end
        else:
            # Find trans between two image plains
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            gray1 = img1
            if len(gray1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, criteria_calib)

            if not ret1:
                return img_warp

            objp = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria_corners)
            objp = objp.squeeze()

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_corners)
        # corners_ud = stereo_mapping.undistort_point(corners2, cam=cam)
        corners_ud = cam_calib.undistort_point(corners2, cam=cam)
        corners_ud = corners_ud.squeeze()

        H, _ = cv2.findHomography(corners_ud, objp)
        print('perspective_transform: ' + print_homo(H))
        H_inv = np.linalg.inv(H)
        H_inv /= H_inv[2, 2]
        print('perspective_trans_inv: ' + print_homo(H_inv))

        # Draw and display the corners
        # img_calib = stereo_mapping.undistort_image(img)
        img_calib = cam_calib.undistort_image(img)
        img_calib = cv2.drawChessboardCorners(img_calib, pattern_size, corners_ud, ret)
        img_warp = cv2.warpPerspective(img_calib, H, warp_size)
        # img_warp = np.zeros_like(img)
        # for point in objp:
        #     img_warp = cv2.drawMarker(img_warp, np.int32(point), (0, 0, 255))

    return img_warp


def t_selection():

    img = cv2.imread('images3/image0.png')
    print(img.shape)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow('image', img)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # draw_world_points(gray, K, D)


def t_chessboard(gray, K, D, T_cw):
    CHECKERBOARD = (6, 9)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return

    xy_undistorted = cv2.undistortPoints(corners, K, D)

    xyz_list = []
    for corner in xy_undistorted:
        corner = np.squeeze(corner)
        u = corner[0]
        v = corner[1]
        XYZ = Triangulation.calculate_XYZ(u, v)
        xyz_list.append(np.squeeze(XYZ) * 3.676874503713879)

    xyz_list = np.array(xyz_list)

    # off-load visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # ax.scatter(xyz_list[:, 0], xyz_list[:, 1], xyz_list[:, 2], marker='.')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # set_axes_equal(ax)
    # plt.show()


def t_stereo_euroc(settings):

    ds_root = settings['ds_root']

    # load cameras
    cam0_path = os.path.join(ds_root, 'cam0', 'data.csv')
    cam0_loader = ImageLoaderEuRoC(cam0_path)
    cam0_calib = CalibPinholeRadTan(settings)
    cam0_calib1 = CalibPinholeRadTan(settings)
    T_bc0 = np.array(settings['T_bc0'], dtype=np.float32).reshape(4, 4)

    cam1_path = os.path.join(ds_root, 'cam1', 'data.csv')
    cam1_loader = ImageLoaderEuRoC(cam1_path)
    cam1_calib = CalibPinholeRadTan(settings)
    T_bc1 = np.array(settings['T_bc1'], dtype=np.float32).reshape(4, 4)
    T_SB = np.linalg.inv(np.array(settings['T_BS'], dtype=np.float32).reshape(4, 4))

    # load camera poses (e.g. from GT file)
    # row format for EuRoC: [ts, tx, ty, tz, qw, qx, qy, qz, ...]
    gt_path = os.path.join(ds_root, 'state_groundtruth_estimate0/data.csv')
    gt_pose = PoseLoader(gt_path, row_format='ts,tx,ty,tz,qw,qx,qy,qz,...')

    # prepare samples
    img0 = cam0_loader.get_image_item(1430)
    img1 = cam1_loader.get_image_item(1430)
    img2 = cam0_loader.get_image_item(1460)

    # These are actually T_RS
    T0_wb = gt_pose.get_pose_ts(img0.ts).T_sr
    # T1_wb = gt_pose.get_pose_ts(img1.ts)
    T2_wb = gt_pose.get_pose_ts(img2.ts).T_sr

    T0_wc0 = T0_wb @ T_SB @ T_bc0
    T1_wc1 = T0_wb @ T_SB @ T_bc1
    T2_wc0 = T2_wb @ T_SB @ T_bc0

    # detect and associate features
    orb_det_match = KptDetMatch(n_kpts=500)
    kp0, des0, kp1, des1, matches = orb_det_match.detect_and_match(img0.frame, img1.frame)
    kp01, des01, kp11, des11, matches1 = orb_det_match.detect_and_match(img0.frame, img2.frame)

    # reconstruct the scene
    cam0_calib.update_pose_trans(np.linalg.inv(T0_wc0))
    cam1_calib.update_pose_trans(np.linalg.inv(T1_wc1))
    pts3d_stereo = Triangulation.reconstruct_kpt(cam0_calib, cam1_calib, kp0, kp1, matches)

    cam0_calib.update_pose_trans(np.linalg.inv(T0_wc0))
    cam0_calib1.update_pose_trans(np.linalg.inv(T2_wc0))
    pts3d_epipolar = Triangulation.reconstruct_kpt(cam0_calib, cam0_calib1, kp01, kp11, matches1)

    # visualization
    # print(len(pts3d_stereo))
    img3 = cv2.drawMatches(img0.frame, kp0, img1.frame, kp1, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches0', img_resize(img3, 0.75))
    cv2.waitKey()
    if len(pts3d_stereo) > 0:
        draw_camera_and_wpts([cam0_calib, cam1_calib], pts3d_stereo)

    img4 = cv2.drawMatches(img0.frame, kp01, img2.frame, kp11, matches1[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches1', img_resize(img4, 0.75))
    cv2.waitKey()
    if len(pts3d_epipolar) > 0:
        draw_camera_and_wpts([cam0_calib, cam0_calib1], pts3d_epipolar)

    cv2.destroyAllWindows()


def t_tum_rgbd(ds_root, settings):
    # load cameras
    cam0_rgb_file = os.path.join(ds_root, 'rgb.txt')
    cam0_depth_file = os.path.join(ds_root, 'depth.txt')
    cam0_loader = ImageLoader(cam0_rgb_file, 'images_file')
    cam0_depth = ImageLoader(cam0_depth_file, 'images_file')
    cam0_calib = CalibPinholeRadTan(settings)

    # load camera poses (e.g. from GT file)
    # row format for TUM-RGBD: [ts, tx, ty, tz, qx, qy, qz, qw]
    gt_path = os.path.join(ds_root, 'groundtruth.txt')
    gt_pose = PoseLoader(gt_path, row_format='ts,tx,ty,tz,qx,qy,qz,qw')

    # DL detector
    ml_detector = ObjectDetectionTflite(settings)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')
    plt.ion()
    plt.show()

    while cam0_loader.is_ok():
        image = cam0_loader.get_next()
        depth = cam0_depth.get_next()

        if image is None:
            print('Null image detected')
            break

        pose = gt_pose.get_pose_ts(image.ts)
        cam0_calib.update_pose_trans(np.linalg.inv(pose.T_sr))

        results = ml_detector.detect(image.frame)

        img_show = np.copy(image.frame)
        max_conf = 0.0
        max_res = []
        Xw = []
        for result in results:
            if result[1] > max_conf:
                max_conf = result[1]
                max_res = result
            cv2.rectangle(img_show, result[2:4], result[4:], (0, 0, 255), 2)
            cv2.putText(img_show, result[0], result[2:4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if len(max_res) > 0:
            min_x, min_y, max_x, max_y = max_res[2:]
            points = get_img_points_roi([min_x, max_x, min_y, max_y], image.frame.shape)
            Xr, tr = cam0_calib.unproject(points)
            S = depth.frame[points[:, 1], points[:, 0]].reshape(len(points), 1)
            SS = S / max(S) * np.linalg.norm(tr[0])
            Xw = Xr * SS + tr
            colors = np.float32(image.frame[points[:, 1], points[:, 0]])
            colors = np.fliplr(colors)
            colors /= np.max(colors)

            ax.scatter(Xw[:, 0], Xw[:, 1], Xw[:, 2], c=colors)
            # stride = 1
            # Xww = Xw.reshape(max_x - min_x, max_y - min_y, 3)
            # ax.plot_surface(Xw[:, 0], Xw[:, 1], Xw[:, 2], rstride=stride, cstride=stride, facecolors=colors)

            # plt.show()
            plt.draw()
            plt.pause(0.001)

        cv2.imshow('TUM-RGB', img_show)
        if depth is not None:
            cv2.imshow('TUM-Depth', depth.frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == 27:
            break


def t_main():
    settings = load_settings('../../config/IAUN.yaml')
    stereo_mapping = StereoMapping(settings)
    obj_detector = ObjectDetection(settings)
    base_calib = os.getenv('DATA_DIR', 'cv_stereo')

    m_cam = MultiCam(base_calib, 'image')

    window_width, window_height = settings['img_size']
    cv2.namedWindow('image0', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image0', window_width, window_height)
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', window_width, window_height)

    # set mouse callback function for window
    cv2.setMouseCallback('image0', mouse_callback, 0)
    cv2.setMouseCallback('image1', mouse_callback, 1)

    perspective_images = [base_calib + '/stereo_image0.png', base_calib + '/stereo_image5.png']

    cnt = 0
    file_offset = 1000
    marker_sz = 10
    marker_th = 2
    h_computed = False
    while True:
        # img0, img1 = m_cam.get_next_img()
        img0 = cv2.imread(perspective_images[1], cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(perspective_images[0], cv2.IMREAD_UNCHANGED)

        if img0 is None or img1 is None:
            break

        # if not h_computed:
        #     t_compute_homography(stereo_mapping)
        #     print("Cam0 to Cam1 image Homography is computed")
        # h_computed = True

        # img2 = t_apriltag_mapping(stereo_mapping, obj_detector, img0)
        # img2 = t_chessboard_mapping(stereo_mapping, img0, img1)
        # t_constant_homography(stereo_mapping, img0, img1)
        # t_matching(img0, img1, True)
        img2 = t_perspective_correction(stereo_mapping, img0[:, :640])

        cv2.imshow('image0', img2)
        cv2.imshow('image1', img1)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Test Structure From Motion:
        Known Params:   Camera poses, camera calibration, image observation matches (e.g. key points),
                        and possibly, pixel depth
        Estimate:       World objects (e.g. world point cloud)
    ''')
    parser.add_argument('config_path', help='path to the configuration file')
    parser.add_argument('--ds_root', help='path to the dataset root (e.g. .../EuRoC/mav0', default='.')
    parser.add_argument('--grid_size', nargs='+', help='grid size (default: (6, 9))', default=[])
    parser.add_argument('--img_path', help='path to the calibration image', default='')
    parser.add_argument('--img1_path', help='path to the calibration image1', default='')
    args = parser.parse_args()

    # load settings
    config_file = args.config_path
    settings = load_settings(config_file)

    grid_size = args.grid_size
    if len(grid_size) <= 0:
        grid_size = (6, 9)
    else:
        grid_size = np.int32(args.grid_size)

    t_stereo_euroc(settings)
    # t_tum_rgbd(args.ds_root, settings)
    # img_show = t_perspective_correction(settings, args.img_path, img1_path=args.img1_path, pattern_size=grid_size)
    # cv2.imshow('Image', img_show)
    # cv2.waitKey()

