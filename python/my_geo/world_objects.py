import argparse
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy.polys.subresultants_qq_zz import final_touches
from scipy.spatial.transform import Rotation as scipyR

sys.path.append('../')
from tools.utils import load_settings
from calib.cam_calib_models import CalibPinholeRadTan, CalibFisheye
import cv_mine.cv_utils as cvu
import mviz.viz_pose as mvp


class WorldObject(object):
    def __init__(self, pt3d=np.zeros((3, 1)), color=(0, 0, 0)):
        self.X = pt3d
        self.color = color

    def render3d(self, ax, marker='.'):
        ax.scatter(*self.X.squeeze(), color=self.color, marker=marker)

    def render2d(self, camera):
        x, depth = camera.project(self.X)
        # return point2d, depth, color
        return x, depth, self.color

    def wireframe3d(self, ax):
        self.render3d(ax)


class Point3D(WorldObject):
    def __init__(self, pt3d=np.zeros((3, 1)), color=(0, 0, 0)):
        super().__init__(pt3d=pt3d, color=color)


class Texture3D(WorldObject):
    def __init__(self, texture, texture_area):
        super().__init__()

        self.texture = texture
        self.texture_area = texture_area
        self.lim_x = []
        self.lim_y = []
        self.X = []
        self.Y = []
        self.K = np.identity(3)

        if texture is None or texture_area is None:
            print('Texture construction: Null texture input')
            return

        n_t_area = len(texture_area)

        if n_t_area >= 2:
            self.lim_x = texture_area[:2]
        if n_t_area >= 4:
            self.lim_y = texture_area[2:4]

        h, w = texture.shape[:2]
        x0, x1 = self.lim_x
        # x units (m) per pixel
        xpp = (x1 - x0) / w
        ypp = xpp
        if len(self.lim_y) == 2:
            y0, y1 = self.lim_y
            ypp = (y1 - y0) / h
        else:
            ly = h * ypp
            y0, y1 = -ly / 2.0, ly / 2.0
            self.lim_y = [y0, y1]

        xr = np.arange(x0, x1, xpp)
        yr = np.arange(y0, y1, xpp)

        self.X, self.Y = np.meshgrid(xr, yr)

        # mapping between world coords and image
        xo = 0.5 * (x0 + x1)
        yo = 0.5 * (y0 + y1)
        self.K = np.array([
            [1. / xpp, 0, -xo / xpp + w * 0.5],
            [0, 1. / ypp, -yo / ypp + h * 0.5],
            [0, 0, 1]
        ], dtype=np.float32)

    def draw_texture(self, ax, get_z, stride=1):
        z = get_z(self.X, self.Y)
        ax.plot_surface(self.X, self.Y, z, rstride=stride, cstride=stride, facecolors=self.texture)

    def get_texture_colors(self, world_points, colors):
        # n_points = len(points)
        # colors = np.zeros((n_points, 3))
        # Z coord == 0 -> this messed up everything!
        wp_homo = np.concatenate((world_points[:, :2], np.ones((len(world_points), 1))), axis=1)
        pts_img = self.K @ wp_homo.transpose()
        pts_img = np.int32(np.round(pts_img.transpose()))
        h, w = self.texture.shape[:2]
        for i, pt in enumerate(pts_img):
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                colors[i] = self.texture[pt[1], pt[0]]
        return colors

    def get_texture_bounds(self):
        x0, x1 = self.lim_x
        y0, y1 = self.lim_y
        return np.array([
            [x0, y0, 0],
            [x1, y0, 0],
            [x1, y1, 0],
            [x0, y1, 0]
        ], dtype=np.float32)


class Plain(WorldObject):
    def __init__(self, pt3d=np.zeros((3, 1)), normal=np.ones((3, 1)),
                 color=(0, 0, 0), texture=None, texture_area=None):
        """
        Plain: Described by Point-Normal, 3-Points, ABCD
        :param pt3d: For plains, self.X is the origin of the plain or point in Point-Normal rep
        :param normal:
        :param points:
        :param color:
        :param texture:
        """
        super().__init__(pt3d=pt3d, color=color)
        self.texture = Texture3D(texture, texture_area)
        self.normal = normal
        self.abcd = self.point_normal_to_abcd(pt3d, normal)

    @staticmethod
    def point_normal_to_abcd(pt3d, normal):
        return np.array([*normal.squeeze(), -np.sum(pt3d * normal)])

    @staticmethod
    def three_points_to_pt_normal(points):
        p0, p1, p2 = points
        p10 = p1 - p0
        p20 = p2 - p0
        normal = np.cross(p10, p20)
        return p0, normal

    @staticmethod
    def three_points_to_abcd(points):
        p0, normal = Plain.three_points_to_pt_normal(points)
        return Plain.point_normal_to_abcd(p0, normal)

    @staticmethod
    def get_default_points():
        return np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32)

    def set_texture(self, texture, texture_area):
        self.texture = Texture3D(texture, texture_area)

    def calc_xyz(self, X, Y):
        a, b, c, d = self.abcd
        return -(a * X + b * Y + d) / (c + 0.00001)


    def intersect_rays(self, Xr, tr):
        a, b, c, d = self.abcd
        S = -(d + a * tr[:, 0] + b * tr[:, 1] + c * tr[:, 2]) / (a * Xr[:, 0] + b * Xr[:, 1] + c * Xr[:, 2])
        SS = np.repeat(S.reshape(-1, 1), 3, axis=1)
        Xw = Xr * SS + tr
        return Xw, S

    @staticmethod
    def compute_depth(Xw, t_wc):
        t_wc_v = np.repeat(t_wc.reshape((1, 3)), len(Xw), axis=0)
        depth_v = np.linalg.norm(Xw - t_wc_v, axis=1)
        return depth_v

    @staticmethod
    def get_valid_points_depth(depth_v, S):
        try:
            valid_depth = np.bitwise_and(depth_v < 1000.0, S > 0)
            dd = depth_v[valid_depth]
            if len(dd) <= 0:
                return [], []
            max_depth = np.max(dd)
            depth_v_norm = 1.0 - depth_v / max_depth
            depth_color = np.repeat(depth_v_norm[valid_depth].reshape(-1, 1), 3, axis=1)
            return valid_depth, depth_color
        except ValueError:
            print('Plain.get_valid_points_depth, ValueError Exception')
            return [], []

    def get_image_bounds(self):
        texture_corners = self.texture.get_texture_bounds()
        # these points are correct because unproject produces the initial inputs
        texture_img_corners, s = camera.project(texture_corners)
        texture_img_corners *= np.sign(s)
        X_tic = texture_img_corners[:, 0]
        Y_tic = texture_img_corners[:, 1]
        x_min = int(min(np.floor(X_tic)))
        x_max = int(max(np.ceil(X_tic)))
        y_min = int(min(np.floor(Y_tic)))
        y_max = int(max(np.ceil(Y_tic)))
        # there's a little offset between texture_img_corners and min/max rectangle

        return [x_min, x_max, y_min, y_max]

    def render2d_global(self, camera):
        # render all image
        w, h = camera.img_size
        image = np.zeros((h, w, 3))
        # create the image pixel mesh
        xyc = cvu.get_img_points_roi([0, w, 0, h], (w, h))
        # unproject image points to get camera rays
        Xr, tr = camera.unproject(xyc)
        if len(Xr) <= 0:
            return image
        # intersect rays with the plain to find the depth and final coords
        Xw, S = self.intersect_rays(Xr, tr)
        # draw the pixels with resulting colors
        # here, change the color based on depth
        depth_v = self.compute_depth(Xw, camera.T_wc[:3, 3])
        valid_depth, depth_color = self.get_valid_points_depth(depth_v, S)
        valid_pts = xyc[valid_depth, :2]
        image[valid_pts[:, 1], valid_pts[:, 0], :] = depth_color * self.color
        # return the image
        return image
        # return point2d, depth, color

    def render2d_texture(self, camera):
        # Although everything is symmetric, the produced image shows an offset in x-y direction???
        w, h = camera.img_size
        image = np.zeros((h, w, 3)) # + self.color
        # determine the texture area
        # texture_area = self.get_image_bounds()
        # create the image pixel mesh
        # xyc = cvu.get_img_points_roi(texture_area, (w, h))
        xyc = cvu.get_img_points_roi([0, w, 0, h], (w, h))
        if len(xyc) <= 0:
            return image
        # unproject image points to get camera rays
        Xr, tr = camera.unproject(xyc)
        if len(Xr) <= 0:
            return image
        # intersect rays with the plain to find the depth and final coords
        Xw, S = self.intersect_rays(Xr, tr)
        # draw the pixels with resulting colors
        # here, change the color based on depth
        depth_v = self.compute_depth(Xw, camera.T_wc[:3, 3])
        valid_depth, depth_color = self.get_valid_points_depth(depth_v, S)
        if len(valid_depth) <= 0:
            return image
        valid_pts = xyc[valid_depth, :2]
        colors = np.ones_like(depth_color) * self.color  # depth_color * self.color
        colors = self.texture.get_texture_colors(Xw[valid_depth], colors)
        image[valid_pts[:, 1], valid_pts[:, 0], :] = colors
        # return the image
        return image

    def render2d(self, camera):
        if self.texture.texture is None:
            return self.render2d_global(camera)
        return self.render2d_texture(camera)

    def render3d(self, ax, pts_xy=None):
        if pts_xy is None:
            pts_xy = self.get_default_points()
        X = pts_xy[:, 0]
        Y = pts_xy[:, 1]
        final_pts = list(zip(X, Y, self.calc_xyz(X, Y)))
        ax.add_collection3d(Poly3DCollection([final_pts], color=self.color))

    def wireframe3d(self, ax, pts_xy=None):
        if pts_xy is None:
            pts_xy = self.get_default_points()
        X = pts_xy[:, 0]
        Y = pts_xy[:, 1]
        closed_shape = list(zip(X, Y, self.calc_xyz(X, Y)))
        closed_shape.append(closed_shape[0])
        closed_shape = np.array(closed_shape, dtype=np.float32)
        for pt in closed_shape:
            ax.scatter(*pt)
        ax.plot(closed_shape[:, 0], closed_shape[:, 1], zs=closed_shape[:, 2], color=self.color)

    def z_render2d_texture_t1(self, image, texture_img_corners, bounds):
        x_min, x_max, y_min, y_max = bounds
        imc = np.int32(texture_img_corners)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (1, 0, 0), 2)
        image = cv2.rectangle(image, imc[0, :2], imc[2, :2], (0, 0, 1), 2)
        # return image
        test_points = np.array([
            [290, 187], [445, 187], [490, 383], [247, 383]
        ])
        Xrr, trr = camera.unproject(test_points)
        Xww, SS = self.intersect_rays(Xrr, trr)


class RectCube(WorldObject):
    def __init__(self):
        super().__init__()


class Cube(WorldObject):

    def __init__(self):
        super().__init__()
        self.cube_points = np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0]]).T
        self.c_arr = np.array([120, 140, 160, 180, 200, 220])

        self.n = np.array([[1.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.],
                      [0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, -1.]])

        self.p0 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])

        res = 100
        res2 = res ** 2
        ax = np.linspace(0.0, 1.0, res)
        p0, p1 = np.meshgrid(ax, ax)
        p0 = p0.reshape((res2, 1))
        p1 = p1.reshape((res2, 1))
        assert len(p0) == len(p1)
        n0 = np.zeros((res2, 1))
        n1 = n0 + 1.0

        X0 = np.concatenate([n0, p0, p1], axis=1)
        X1 = np.concatenate([n1, p0, p1], axis=1)
        Y0 = np.concatenate([p0, n0, p1], axis=1)
        Y1 = np.concatenate([p0, n1, p1], axis=1)
        Z0 = np.concatenate([p0, p1, n0], axis=1)
        Z1 = np.concatenate([p0, p1, n1], axis=1)

        self.Pw = np.concatenate([X0, X1, Y0, Y1, Z0, Z1])

        colors = np.array([120, 140, 160, 180, 200, 220])
        self.colors = np.repeat(colors, len(X0)).reshape((len(self.Pw), 1))

    @staticmethod
    def get_points_on_cube(p):
        c0 = (0.0 <= p) & (p <= 1.0)
        return c0[:, 0] & c0[:, 1] & c0[:, 2]

    # A unit cube is defined from (0, 0, 0) to (1, 1, 1)
    def get_cube_color(self, P):
        pts_on_plains = abs(np.sum((P.reshape(1, 3) - self.p0) * self.n, axis=1)) < 1e-9

        if pts_on_plains.any():
            return self.c_arr[pts_on_plains][0]
        else:
            return -1

    def get_cube_image_bounds(self, R_cw, t_cw, K):

        # Find corresponding in camera
        p_c = K @ (R_cw @ self.cube_points + t_cw)
        p_c = p_c / p_c[2, :]
        max_xy = np.max(p_c[0:2, :], axis=1)
        min_xy = np.min(p_c[0:2, :], axis=1)

        return np.concatenate((min_xy, max_xy))

    def get_cube_color_enhanced(self, l, l0):

        l0 = l0.reshape((1, 3))
        l = l.reshape((1, 3))
        l = l / np.linalg.norm(l)
        ll = np.repeat(l, 6, axis=0)

        # d = (p0 - l0) . n / l . n
        b = np.sum(ll * self.n, axis=1)
        a = np.sum((self.p0 - l0) * self.n, axis=1)
        d = a / b

        # p = l0 + ld
        dd = np.repeat(d.reshape((6, 1)), 3, axis=1)
        p = l0 + ll * dd

        cond = self.get_points_on_cube(p)

        if cond.any():
            # pp = p[cond, :] - l0
            # d_p = np.sum(pp * pp, axis=1)
            theta = np.arccos(b[cond])
            return self.c_arr[cond][np.argmin(theta)]
        else:
            return -1

    def render2d(self, camera):

        img_w, img_h = camera.img_size

        image = np.ones((img_h, img_w)) * 255

        minmax_xy = self.get_cube_image_bounds(camera.R_cw, camera.t_cw.reshape((3, 1)), camera.K)
        max_x = min(img_w, minmax_xy[2])
        min_x = max(0.0, minmax_xy[0])
        max_y = min(img_h, minmax_xy[3])
        min_y = max(0.0, minmax_xy[1])

        vx = range(0, img_w)
        vy = range(0, img_h)

        x_cube = [v for v in vx if min_x <= v <= max_x]
        y_cube = [v for v in vy if min_y <= v <= max_y]

        for y in y_cube:
            for x in x_cube:
                p_c = np.array([[x, y]])
                l, t_wc = camera.unproject(p_c)
                c = self.get_cube_color_enhanced(l, t_wc)
                if c > 0:
                    image[y, x] = c

        # plt.imshow(image, interpolation='nearest')
        # plt.show()
        return image

    def take_picture_pinhole_distorted(self, camera):

        img_w, img_h = camera.img_size

        image = np.zeros((img_h, img_w))

        pw = self.Pw
        Pw = np.concatenate([pw, np.ones((pw.shape[0], 1))], axis=1).T

        Pc = camera.T_cw[0:3, :] @ Pw

        colors = self.colors

        margin = 5

        for i in range(0, Pc.shape[1]):
            pc = Pc[:, i]
            z0 = pc[2]
            pc_norm = pc / z0
            pc_p = camera.distort_radial_tangential(pc_norm)
            pc_p = np.array([pc_p[0], pc_p[1], 1.0]).reshape((3, 1))
            xy_p = camera.K @ pc_p
            if not (margin <= xy_p[0][0] <= img_w - margin and margin <= xy_p[1][0] <= img_h - margin):
                continue
            X, Y, I = cvu.get_intensity_gaussian(colors[i], np.array([xy_p[1], xy_p[0]]).squeeze(), 0.33)
            try:
                image[X, Y] += I
            except:
                print('array over range access error: (x, y) = (%.2f, %.2f)' % (xy_p[0][0], xy_p[1][0]))

        return image

    def take_picture_fisheye(self, camera):

        img_w, img_h = camera.img_size

        image = np.zeros((img_h, img_w))

        pw = self.Pw
        Pw = np.concatenate([pw, np.ones((pw.shape[0], 1))], axis=1).T

        colors = self.colors
        mg = 5

        for i in range(0, Pw.shape[1]):
            xy_p = camera.project(Pw[:, i])

            if not camera.is_in_image(xy_p, margin=mg):
                continue
            X, Y, I = cvu.get_intensity_gaussian(colors[i], np.array([xy_p[1], xy_p[0]]), 0.33)
            try:
                image[X, Y] += I
            except:
                print('array over range access error: (x, y) = (%.2f, %.2f)' % (xy_p[0][0], xy_p[1][0]))

        return image

    def render3d(self, ax, marker='.'):
        w_cube_x = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        w_cube_y = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        w_cube_z = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])

        # mpv.draw_frame(ax, world_frame['origin'], world_frame["orientation"], scale=2)
        # mpv.draw_frame(ax, cam0.T_wc[:3, 3], cam0.T_wc[:3, :3], scale=0.5)
        # mpv.draw_frame(ax, cam1.T_wc[:3, 3], cam1.T_wc[:3, :3], scale=0.5)
        ax.plot(w_cube_x, w_cube_y, w_cube_z)

        ax.set_aspect('equal', adjustable='box')
        # return ax
        # plt.show()


class Line3D(WorldObject):
    def __init__(self):
        super().__init__()


class Curve3D(WorldObject):
    def __init__(self):
        super().__init__()


class Sphere(WorldObject):
    def __init__(self):
        super().__init__()


class Cylinder(WorldObject):
    def __init__(self):
        super().__init__()


class Cone(WorldObject):
    def __init__(self):
        super().__init__()


class MeshCell(WorldObject):
    def __init__(self):
        super().__init__()


def t_plain(img_texture, camera):

    xy_orig = np.zeros((3, 1))
    xy_normal = np.array([0, 0, 1]).reshape((3, 1))
    plain_xy = Plain(pt3d=xy_orig, normal=xy_normal, color=(1, 1, 0),
                     texture=img_texture, texture_area=[-1.0, 1.0])

    ax = plt.figure().add_subplot(projection='3d')

    plain_xy.render3d(ax, pts_xy=plain_xy.texture.get_texture_bounds())
    mvp.draw_frame(ax, xy_orig.flatten(), np.eye(3))
    mvp.draw_frame(ax, camera.T_wc[:3, 3], camera.T_wc[:3, :3])
    mvp.draw_cam_frustum(ax, camera)
    plain_xy.wireframe3d(ax)
    if plain_xy.texture is not None:
        plain_xy.texture.draw_texture(ax, plain_xy.calc_xyz, stride=8)

    ax.set_aspect('equal', adjustable='box')
    plt.show()

    for i in range(30):
        image = plain_xy.render2d(camera)
        cv2.imshow("Street", np.uint8(255 * image))
        cv2.waitKey(1)
        camera.update_pose_inv(camera.T_wc[:3, :3], camera.T_wc[:3, 3] + np.array([0, -0.1, 0]))

    # image = plain_xy.render2d(camera)
    # cv2.imshow('Rendered', image)
    # cv2.waitKey()


def t_cube(camera):
    my_cube = Cube()

    image = my_cube.render2d(camera)
    # image = my_cube.take_picture_pinhole_distorted(camera)
    # image = my_cube.take_picture_fisheye(camera)

    cv2.imshow("Cube Image", np.uint8(image))
    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Work with 3D Geometrical Shapes and Objects
    ''')
    parser.add_argument('--path_texture', help='path to the texture image', default=None)
    parser.add_argument('--path_settings', help='path to the cam clib settings', default=None)
    args = parser.parse_args()

    settings = None
    camera = None
    img_texture = None
    if args.path_settings is not None:
        settings = load_settings(args.path_settings)
        dist_type = settings['distortion_type']
        if dist_type == 'equidistant':
            camera = CalibFisheye(settings)
        else:
            camera = CalibPinholeRadTan(settings)
        camera.update_pose_inv(scipyR.from_euler('zyx', np.array([0.0, -180.0, -50.0]), degrees=True).as_matrix(),
                               np.array([0.0, 1.5, 0.2]))
    if args.path_texture is not None:
        img_texture = plt.imread(args.path_texture)

    t_plain(img_texture, camera)
    # t_cube(camera)


