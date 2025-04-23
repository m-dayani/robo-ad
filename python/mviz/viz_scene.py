import argparse

import matplotlib.pyplot as plt
import numpy as np

from viz_pose import plot_traj_3d, draw_coord_sys, draw_frame


def filter_map(mapData, xlim=None, ylim=None, zlim=None):

    nMapData = len(mapData)

    xMask = np.ones(nMapData, dtype=bool)
    if xlim is not None and len(xlim) == 2:
        xMask = np.logical_and(mapData[:, 0] >= xlim[0], mapData[:, 0] < xlim[1])
    yMask = np.ones(nMapData, dtype=bool)
    if ylim is not None and len(ylim) == 2:
        yMask = np.logical_and(mapData[:, 1] >= ylim[0], mapData[:, 1] < ylim[1])
    zMask = np.ones(nMapData, dtype=bool)
    if zlim is not None and len(zlim) == 2:
        zMask = np.logical_and(mapData[:, 2] >= zlim[0], mapData[:, 2] < zlim[1])

    mapMask = np.logical_and(np.logical_and(xMask, yMask), zMask)
    mapData = mapData[mapMask, :]

    return mapData, nMapData - np.sum(mapMask)


def plot_pc_3d(ax, map_pc, style_):
    """
    style_: [marker color, marker sign]
    """
    ax.scatter(map_pc[:, 0], map_pc[:, 1], map_pc[:, 2], c=style_[0], marker=style_[1])


def draw_pose_map(poseData, mapData, xlim=None, ylim=None, zlim=None):

    # Filter data (fit to bounds)
    mapData, nOutliers = filter_map(mapData, xlim, ylim, zlim)
    print("Num. map point outliers: %d" % (nOutliers))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot
    plot_pc_3d(ax, mapData, ['g', '.'])

    plot_traj_3d(ax, poseData[:, 1:4], 'pose', ['y', '*', 'b'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.legend()

    plt.show()


def draw_camera_and_wpts(cam_objs, pts_w, ws=1.0):
    ax = plt.figure().add_subplot(projection='3d')

    cs = 3.0
    for i, cam_obj in enumerate(cam_objs):
        label = 'C' + str(i)
        # draw_coord_sys(ax, cam_obj.T_cw, sc=cs, label=label)
        draw_frame(ax, cam_obj.T_wc[0:3, 3], cam_obj.T_wc[0:3, 0:3], label=label, scale=cs)

    ss = 5.0
    draw_frame(ax, np.zeros(3), np.identity(3), scale=ss, label='W')

    map_pc, _ = filter_map(np.array(pts_w))
    plot_pc_3d(ax, map_pc, ('r', '.'))

    ax.set_aspect('equal', adjustable='box')

    plt.show()


def t_draw_pose_map():
    defPathMap = '../../../data/orb_map.txt'
    defPathPose = '../../results/ev_ethz/okf_ev_ethz_mono_ev_im_slider_depth_etm1_ftdtm2_tTiFr_0.44483.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots estimated estimated camera pose and 3D world points. 
    ''')
    parser.add_argument('-path_pose', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=defPathPose)
    parser.add_argument('-path_map', help='estimated map (format: x y z)', default=defPathMap)
    parser.add_argument('-xlim', help='x limits: [low up]', nargs=2, type=float, default=[])
    parser.add_argument('-ylim', help='y limits: [low up]', nargs=2, type=float, default=[])
    parser.add_argument('-zlim', help='z limits: [low up]', nargs=2, type=float, default=[])

    args = parser.parse_args()

    # Load Pose Data (ts1 x1 y1 z1 i1 j1 k1 w1\n...)
    poseData = np.loadtxt(args.path_pose)

    # Load Map Data (x1 y1 z1\nx2 y2 ...)
    mapData = np.loadtxt(args.path_map)

    draw_pose_map(poseData, mapData, args.xlim, args.ylim, args.zlim)


if __name__ == "__main__":
    t_draw_pose_map()

