"""
    Visualization Options:
        - matplotlib
        - Open3D
        - OpenCV
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../data_loader/')
from data_loader.pose_loader import PoseLoader


def draw_frame(ax, p_nb, r_nb, label='', scale=1.0, linewidth='1.5'):

    # rotate frame n (Identity mat) to align it with the body frame
    b = scale * r_nb.T + p_nb
    b = b.T

    ox = p_nb[0]
    oy = p_nb[1]
    oz = p_nb[2]
    dd = 0.0 * scale * np.ones(p_nb.shape)
    rr = np.matmul(r_nb.T, dd) + p_nb

    ax.plot([ox, b[0, 0]], [oy, b[1, 0]], [oz, b[2, 0]], color='r', linewidth=linewidth)
    ax.plot([ox, b[0, 1]], [oy, b[1, 1]], [oz, b[2, 1]], color='g', linewidth=linewidth)
    ax.plot([ox, b[0, 2]], [oy, b[1, 2]], [oz, b[2, 2]], color='b', linewidth=linewidth)

    if len(label) > 0:
        ax.text(rr[0], rr[1], rr[2], label)


def draw_coord_sys(ax, trans, sc=1.0, label=''):

    coords = np.identity(4, dtype=np.float32)
    coords[3, 0:3] = np.array([1, 1, 1])

    new_coords = np.matmul(trans, coords)

    x = np.repeat(new_coords[0, 3], 3)
    y = np.repeat(new_coords[1, 3], 3)
    z = np.repeat(new_coords[2, 3], 3)

    u = new_coords[0, 0:3]
    v = new_coords[1, 0:3]
    w = new_coords[2, 0:3]

    cols = ['red', 'green', 'blue']

    for i in range(3):
        ax.quiver(x[i], y[i], z[i], u[i], v[i], w[i], length=sc, normalize=True, colors=cols[i])

    dtext = 0.08 * sc
    ax.text(x[0]-dtext, y[0]-dtext, z[0]-dtext, label, color='m')


def draw_cam_frustum(ax, camera, scale=0.5):
    img_w, img_h = camera.img_size

    # Remember your frames are represented as T_wc, so you need to invert it
    T_wc = camera.T_wc
    R_wc0 = T_wc[:3, :3]
    t_wc0 = T_wc[:3, 3]
    K_1 = camera.K_1

    cam_points = np.array([[0, 0, 1.0],
                           [img_w, 0, 1.0],
                           [img_w, img_h, 1.0],
                           [0, img_h, 1.0],
                           [0, 0, 1.0],
                           [0, 0, 0.0],
                           [img_w, img_h, 1.0],
                           [0, img_h, 1.0],
                           [0, 0, 0.0],
                           [img_w, 0, 1.0]]).T

    cc = scale * R_wc0 @ K_1 @ cam_points + t_wc0.reshape((3, 1))

    ax.plot(cc[0, :], cc[1, :], cc[2, :])


def draw_vector(ax, orig, v, label='', scale=1.0, c=''):

    vv = np.c_[orig, orig + v * scale].T

    if len(c) > 0:
        ax.plot(vv[:, 0], vv[:, 1], vv[:, 2], c=c)
    else:
        ax.plot(vv[:, 0], vv[:, 1], vv[:, 2])

    if len(label) > 0:
        ax.text(vv[1, 0], vv[1, 1], vv[1, 2], label)


def draw_pose(t, position, orientation):

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')

    ax.plot(position[:, 0], position[:, 1], position[:, 2])

    for i in range(0, int(len(t)/2), 50):
        draw_frame(ax, position[i], orientation[i], 'O'+str(i), 0.3)

    draw_frame(ax, np.zeros(3), np.eye(3), 'Ref', 1.0)

    plt.show()


def comp_orientations(rot_mat_list, tags):

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')

    for i in range(len(rot_mat_list)):
        draw_frame(ax, np.array([0.0, 0.0, 0.0]), rot_mat_list[i], tags[i])

    plt.show()


def plot_traj_3d(ax, traj_data, label_, style_):
    """
    :param style_: [marker color, marker sign, line color]
    """
    ax.scatter(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], c=style_[0], marker=style_[1])
    ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], color=style_[2], label=label_)


def draw_pose_gt_est(gtData, estDataAligned):

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')

    plot_traj_3d(ax, gtData, 'gt traj.', ['y', '*', 'b'])

    plot_traj_3d(ax, estDataAligned, 'est. traj.', ['c', '*', 'r'])

    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def draw_line_3d(ax, start, end, color='b'):
    # draw trajectory between two pose centers
    ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=color)


def draw_pose_arr(ax, pose_arr, pose_step=1, frame_step=10, frame_scale=0.1):

    last_orig = None
    frame_cnt = 0
    pose_cnt = 0
    for pose in pose_arr:
        # EuRoC poses are recored as T_RS_R
        T_rs = pose.T_sr
        orig = T_rs[0:3, 3]
        R = T_rs[0:3, 0:3]
        if frame_cnt % frame_step == 0:
            draw_frame(ax, orig, R, scale=frame_scale)
        show_pose = pose_cnt % pose_step == 0
        if last_orig is not None and show_pose:
            draw_line_3d(ax, last_orig, orig)
        if show_pose:
            last_orig = orig
        frame_cnt += 1
        pose_cnt += 1


def t_coord_frame():
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='box')
    # draw_frame(ax, np.zeros(3), np.identity(3))
    draw_coord_sys(ax, np.identity(4), label='W')
    plt.show()


def t_draw_pose(pose_file, n_pose, pose_step, frame_step, frame_scale, pose_format='ts,tx,ty,tz,qw,qx,qy,qz,...'):
    # euroc pose estimates
    gt_pose = PoseLoader(pose_file, row_format=pose_format)

    pose_arr = []
    for i in range(n_pose):
        pose = gt_pose.get_next()
        if pose is not None:
            pose_arr.append(pose)

    ax = plt.figure().add_subplot(projection='3d')

    draw_frame(ax, np.zeros(3), np.identity(3))
    draw_pose_arr(ax, pose_arr, pose_step, frame_step, frame_scale)

    ax.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Drawing 3D poses of robots, their relations, and the trajectory
    ''')
    parser.add_argument('path', help='path to the pose file')
    args = parser.parse_args()

    # t_coord_frame()
    t_draw_pose(args.path, 10000, 10, 100, 0.05)

