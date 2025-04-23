import os
import sys
import argparse
import pickle

import cv2
import numpy as np

sys.path.append('../')
from tools.utils import load_settings


class CalibPinhole:
    def __init__(self, settings):

        self.img_size = (640, 480)
        self.intrinsics = [320, 320, 320, 240]
        if settings is not None:
            # image size/resolution
            self.img_size = settings['img_size']
            # intrinsics
            self.intrinsics = settings['intrinsics']

        self.fx = self.intrinsics[0]
        self.fy = self.intrinsics[1]
        self.cx = self.intrinsics[2]
        self.cy = self.intrinsics[3]
        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0., 0., 1.]])
        self.K_1 = np.linalg.inv(self.K)

        # camera pose
        self.R_cw = np.eye(3)
        self.t_cw = np.zeros((3, 1))
        self.T_cw = np.eye(4)
        self.T_cw[0:3, 0:3] = self.R_cw
        self.T_cw[0:3, 3] = self.t_cw.reshape((3,))
        self.T_wc = np.linalg.inv(self.T_cw)

        self.calibrated = True
        # self.distorted = False

    def update_pose(self, R_cw, t_cw):
        self.R_cw = np.copy(R_cw)
        self.t_cw = np.copy(t_cw)
        self.T_cw[0:3, 0:3] = self.R_cw
        self.T_cw[0:3, 3] = self.t_cw.reshape((3,))
        self.T_wc = np.linalg.inv(self.T_cw)

    def update_pose_inv(self, R_wc, t_wc):
        self.R_cw = R_wc.T
        self.T_wc[:3, :3] = R_wc
        self.T_wc[:3, 3] = t_wc
        self.T_cw = np.linalg.inv(self.T_wc)
        self.t_cw = self.T_cw[:3, 3]

    def update_pose_trans(self, T_cw):
        self.update_pose(T_cw[0:3, 0:3], T_cw[0:3, 3])

    def update_intrinsics(self, K):
        self.K = K
        self.K_1 = np.linalg.inv(K)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    @staticmethod
    def eu2homo(pts):
        """
        Euler to Homogenous Transformation
        :param pts: (N, m)
        :return: (N, m+1)
        """
        return np.concatenate((pts, np.ones((len(pts), 1))), axis=1)

    @staticmethod
    def get_point_h(pt):
        return np.array([pt[0], pt[1], 1.0]).reshape((3, 1))

    @staticmethod
    def homo2eu(pt):
        pt_norm = pt / pt[-1]
        return pt_norm[:-1]

    def project(self, pts_w):
        """
        Project world points to image
        :param pts_w: World points (vector of points (N, 3))
        :return: image points: (N, 2), Depth: (N, 1)
        """
        assert len(pts_w.shape) == 2, 'Calib.project: shape of Pw should be like (N, 3) not ' + str(pts_w.shape)
        Pw = self.eu2homo(pts_w)
        Pc = self.K @ self.T_cw[0:3, :] @ Pw.transpose()
        Pc = Pc.transpose()
        S = np.copy(Pc[:, -1]).reshape(-1, 1)
        Pc /= S
        return Pc[:, :2], S

    def unproject(self, pts_img):
        """
        Unproject image points to world coord sys
        :param pts_img: (N, 2)
        :return: Ray: Xr (N, 3), tr (N, 3) -> s * Xr + tr = Pw
        """
        assert len(pts_img.shape) == 2, 'Calib.unproject: shape of (xy) should be like (N, 2) not ' + str(pts_img.shape)
        Pc = self.eu2homo(pts_img)
        Xr = self.R_cw.T @ Pc.transpose()
        tr = - self.R_cw.T @ np.repeat(self.t_cw.reshape(1, 3), len(pts_img), axis=0).transpose()
        # Pw = self.R_cw.T @ (Pc - self.t_cw.reshape((3, 1)))
        return Xr.transpose(), tr.transpose()

    def save_params(self, file_name):
        with open(file_name, 'wb') as cam_file:
            pickle.dump({'K': self.K, 'T_cw': self.T_cw, 'calibrated': self.calibrated,
                         'img_size': self.img_size}, cam_file)

    def load_params(self, file_name):
        if not os.path.exists(file_name):
            print('params file doesn\'t exist: ' + file_name)
            return
        with open(file_name, 'rb') as cam_file:
            obj = pickle.load(cam_file)

            self.calibrated = obj['calibrated']

            if self.calibrated:
                T_cw = obj['T_cw']
                self.update_pose(T_cw[0:3, 0:3], T_cw[0:3, 3])
                self.update_intrinsics(obj['K'])

    def get_camera_matrix(self):
        return self.K @ self.T_cw[0:3, :]

    def print(self):
        intrinsics = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float32)
        print('intrinsics: [' + ', '.join(map(str, intrinsics)) + ']')


class CalibPinholeRadTan(CalibPinhole):
    def __init__(self, settings):
        super().__init__(settings)

        self.dist_type = 'rad-tan'
        self.D = np.zeros(5)

        if settings is not None:
            # radial-tangential distortion
            self.D = np.array(settings['distortion'], dtype=np.float32)
            self.dist_type = settings['distortion_type']

    def update_distortion(self, D):
        self.D = D

    # def unproject(self, pt_img):
    #     if len(pt_img) < 2:
    #         return None
    #     upts = cv2.undistortPoints(np.array([[pt_img[0], pt_img[1]]], dtype=np.float32), self.K, self.D)
    #     upts = upts.squeeze()
    #     Pc = np.array([upts[0], upts[1], 1.0]).reshape((3, 1))
    #     Pw = self.R_cw.T @ (Pc - self.t_cw.reshape((3, 1)))
    #     return Pw

    def unproject(self, pts_img):
        """
        Unproject image points to world coord sys
        :param pts_img: (N, 2)
        :return: Ray: Xr (N, 3), tr (N, 3) -> s * Xr + tr = Pw
        """
        assert len(pts_img.shape) == 2, 'Calib.unproject: shape of (xy) should be like (N, 2) not ' + str(pts_img.shape)
        N = len(pts_img)
        if N == 0:
            return [], []
        upts = cv2.undistortPoints(np.array(pts_img, dtype=np.float32), self.K, self.D)
        upts = upts.squeeze()
        if N == 1:
            upts = upts.reshape((1, 2))
        Pc = self.eu2homo(upts)
        Pr = self.R_cw.T @ Pc.transpose()
        tr = np.repeat(self.T_wc[:3, 3].reshape((1, 3)), len(Pc), axis=0).transpose()
        # Pw = s * Pr + tr
        return Pr.transpose(), tr.transpose()

    def distort_radial_tangential(self, p):
        # x and y are normalized image coordinates
        x = float(p[0])
        y = float(p[1])
        r2 = x ** 2 + y ** 2

        D = self.D
        k1 = D[0]
        k2 = D[1]
        p1 = D[2]
        p2 = D[3]

        r4 = r2 ** 2
        xp = x * (1 + k1 * r2 + k2 * r4) + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        yp = y * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        return np.array([xp, yp])

    def save_params(self, file_name):
        with open(file_name, 'wb') as cam_file:
            pickle.dump({'K': self.K, 'D': self.D, 'T_cw': self.T_cw, 'calibrated': self.calibrated,
                         'img_size': self.img_size}, cam_file)

    def load_params(self, file_name):
        if not os.path.exists(file_name):
            print('params file doesn\'t exist: ' + file_name)
            return
        with open(file_name, 'rb') as cam_file:
            obj = pickle.load(cam_file)

            self.calibrated = obj['calibrated']

            if self.calibrated:
                self.D = obj['D']
                T_cw = obj['T_cw']
                self.update_pose(T_cw[0:3, 0:3], T_cw[0:3, 3])
                self.update_intrinsics(obj['K'])

    def undistort_point(self, xc, cam=0):
        if len(xc.shape) < 2:
            xc = np.array([xc])
        if cam == 0:
            return cv2.undistortPoints(xc, self.K, self.D, None, self.K).squeeze()
        # else:
        #     return cv2.undistortPoints(xc, self.K1, self.D1, None, self.K1).squeeze()
        return xc

    def normalize_points(self, pt_list, cam=0):
        if cam == 0:
            K = self.K0
        else:
            K = self.K1
        K_1 = np.linalg.inv(K)
        norm_image_points = []
        for point in pt_list:
            new_pt = K_1 @ self.get_point_h(point)
            new_pt = new_pt.reshape((1, 3)).squeeze()
            norm_image_points.append(new_pt[:2])
        return np.array(norm_image_points, dtype=np.float32)

    def undistort_image(self, img, cam=0):
        if cam == 0:
            return cv2.undistort(img, self.K, self.D)
        # else:
        #     return cv2.undistort(img, self.K1, self.D1)
        return img

    def distort_point(self, pt_ud, cam=0):
        if cam == 0:
            D = self.D0
            K = self.K0
        else:
            K = self.K1
            D = self.D1

        p = np.linalg.inv(K) @ self.get_point_h(pt_ud)
        p = p.reshape((1, 3)).squeeze()

        x = float(p[0])
        y = float(p[1])
        r2 = x ** 2 + y ** 2

        k1 = D[0]
        k2 = D[1]
        p1 = D[2]
        p2 = D[3]

        r4 = r2 ** 2
        xp = x * (1 + k1 * r2 + k2 * r4) + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        yp = y * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        # obviously this is wrong, after distortion you can't use a linear trans on non-linear coords!
        pd = K @ np.array([xp, yp, 1.0]).reshape((3, 1))
        pd = pd.reshape((1, 3)).squeeze()

        return pd[:2]

    def print(self):
        super().print()
        D = np.float32(self.D)
        print('distortion: [' + ', '.join(map(str, D)) + ']')


class CalibFisheye(CalibPinholeRadTan):
    def __init__(self, settings):
        super().__init__(settings)

    def is_in_image(self, xy_p, margin=0):
        img_w, img_h = self.img_size
        return margin <= xy_p[0][0] <= img_w - margin and margin <= xy_p[1][0] <= img_h - margin

    def project(self, pts_w):
        """
        Project world points to image
        :param pts_w: World points (vector of points (N, 3))
        :return: image points: (N, 2), Depth: (N, 1)
        """
        assert len(pts_w.shape) == 2, 'Calib.project: shape of Pw should be like (N, 3) not ' + str(pts_w.shape)
        k1, k2, k3, k4 = self.D
        Pw = self.eu2homo(pts_w)
        Pc = self.T_cw[0:3, :] @ Pw.transpose()
        pc = Pc.transpose()
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(r, z)
        d_theta = theta + k1 * (theta ** 3) + k2 * (theta ** 5) + k3 * (theta ** 7) + k4 * (theta ** 9)

        # xy_p = np.ones((3, 1))
        xy_p0 = d_theta * x / r
        xy_p1 = d_theta * y / r
        xy_p = np.vstack((xy_p0, xy_p1, np.ones(len(xy_p0))))

        xy_p = self.K @ xy_p
        xy_p = xy_p.transpose()
        s = np.copy(xy_p[:, 2])
        xy_p /= s.reshape(-1, 1)

        return xy_p, s


class LabCamera(CalibPinholeRadTan):

    def __init__(self, port, settings):
        super().__init__(settings)
        self.port = port
        self.img_size = settings['img_size']
        self.img_width = self.img_size[0]
        self.img_height = self.img_size[1]
        self.capture_duration = 10
        self.usb_cam = None
        # self.init()

        self.H = np.eye(3)
        self.H_1 = np.eye(3)
        self.Hp = self.K @ self.H
        self.Hp_1 = np.linalg.inv(self.Hp)

        self.ax_x_img = []
        self.ax_y_img = []
        self.img_size = np.array([640, 480])

    def update_homography(self):
        try:
            self.H[:, 0:2] = self.T_cw[0:3, 0:2]
            self.H[:, 2] = self.T_cw[0:3, 3]
            self.Hp = self.K @ self.H
            self.H_1 = np.linalg.inv(self.H)
            self.Hp_1 = np.linalg.inv(self.Hp)
        except np.linalg.LinAlgError:
            print('WARNING: Cannot calculate H^-1 because H is singular')

    def project_homo(self, pt_w):
        if len(pt_w) < 2:
            return None
        pt_w = np.array([pt_w[0], pt_w[1], 1.0]).reshape((3, 1))
        xy_p = self.Hp @ pt_w
        depth = np.copy(xy_p[2, :])
        xy_p /= depth
        return np.array(xy_p[0], xy_p[1])

    def unproject_homo(self, pt_img):
        if len(pt_img) < 2:
            return None
        upts = cv2.undistortPoints(np.array([pt_img[0], pt_img[1]], dtype=np.float32), self.K, self.D)
        upts = upts.squeeze()
        upts_homo = np.array([upts[0], upts[1], 1.0])
        pc = self.H_1 @ upts_homo.reshape((3, 1))
        s_pc = np.copy(pc[2, :])
        pc /= s_pc
        return np.array([pc[0], pc[1]])

    def save_params(self, file_name):
        with open(file_name, 'wb') as cam_file:
            pickle.dump({'K': self.K, 'D': self.D, 'T_cw': self.T_cw, 'calibrated': self.calibrated,
                         'ax_x_img': self.ax_x_img, 'ax_y_img': self.ax_y_img, 'img_size': self.img_size}, cam_file)

    def load_params(self, file_name):
        if not os.path.exists(file_name):
            return
        with open(file_name, 'rb') as cam_file:
            obj = pickle.load(cam_file)

            self.calibrated = obj['calibrated']

            if self.calibrated:
                self.D = obj['D']
                T_cw = obj['T_cw']
                self.update_pose(T_cw[0:3, 0:3], T_cw[0:3, 3])
                self.update_intrinsics(obj['K'])
                self.update_homography()
                self.ax_x_img = obj['ax_x_img']
                self.ax_y_img = obj['ax_y_img']
                self.img_size = obj['img_size']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Loads calibration settings from settings file
    ''')
    parser.add_argument('path', help='Path to settings file')
    args = parser.parse_args()

    settings = load_settings(args.path)
    cam_calib = CalibPinholeRadTan(settings)
    print(cam_calib.dist_type)
    print(cam_calib.unproject(np.array([200, 200])))
