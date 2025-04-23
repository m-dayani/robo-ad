import sys

import numpy as np
from scipy.spatial.transform import Rotation as scipyR
import matplotlib.pyplot as plt


# Don't show things in this file (will cause circular dependency, do it in viz_pose)
# sys.path.append('../mviz')
# import viz_pose as vizp


def get_left_q(v):
    qL = np.zeros((4, 4))
    qL[0, 1:] = -v.T
    qL[1:, 0] = v
    qL[1:, 1:] = skew(v)

    return qL


def get_right_q(v):
    qR = np.zeros((4, 4))
    qR[0, 1:] = -v.T
    qR[1:, 0] = v
    qR[1:, 1:] = -skew(v)

    return qR


def rotvec2quat(eta):
    curr_orientation = scipyR.from_rotvec(eta)
    return curr_orientation.as_quat()


def rot2quat(R):
    rObj = scipyR.from_matrix(R)
    return rObj.as_quat()


def rot2rotvec(R):
    rObj = scipyR.from_matrix(R)
    return rObj.as_rotvec()


def rotvec2rot(rv):
    rObj = scipyR.from_rotvec(rv)
    return rObj.as_matrix()


def quat_dot(q1, q2):
    q1_obj = scipyR.from_quat(q1)
    q2_obj = scipyR.from_quat(q2)
    q_res = q1_obj * q2_obj

    return q_res.as_quat()


def quat2rot(q):
    q_obj = scipyR.from_quat(q)
    return q_obj.as_matrix()


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=np.float64)


def exp_q(eta):
    curr_orientation = scipyR.from_rotvec(eta)
    return curr_orientation.as_quat()


def quat_rmat(q):
    q0 = q[0]
    qv = q[1:]

    qrmat = np.zeros((4, 4), dtype=np.float64)
    qrmat[0, 0] = q0
    qrmat[0, 1:] = -qv.transpose()
    qrmat[1:, 0] = qv
    qrmat[1:, 1:] = q0 * np.identity(3, dtype=np.float64) - skew(qv)

    return qrmat


def quat_lmat(q):
    q0 = q[0]
    qv = q[1:]

    qlmat = np.zeros((4, 4), dtype=np.float64)
    qlmat[0] = q0
    qlmat[0, 1:] = -qv.transpose()
    qlmat[1:, 0] = qv
    qlmat[1:, 1:] = q0 * np.identity(3, dtype=np.float64) + skew(qv)

    return qlmat


def exp_q_diff():
    return np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float64)


# TODO: ???? This might be wrong!
def rot_quat_diff(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    return 2 * np.array([[2 * q0 + q3 - q2, 2 * q1 + q2 + q3, 0. + q1 - q0, 0. + q0 + q1],
                         [-q3 + 2 * q0 + q1, q2 + 0. + q0, q1 + 2 * q2 + q3, -q0 + 0. + q2],
                         [q2 - q1 + 2 * q0, q3 - q0 + 0., q0 + q3 + 0., q1 + q2 + 2 * q3]],
                        dtype=np.float64)


def qc(q):
    qq = q
    qq[0:3] *= -1
    return qq


def get_random_pose(R, p):
    if R is None:
        R = scipyR.random(1, random_state=2342345)
    p0 = np.identity(4)
    p0[0:3, 0:3] = R.as_matrix()[0]
    if p is None:
        p = np.random.random(3)
    p0[0:3, 3] = np.array(p).transpose()

    return p0


def interp_pose(p0, p1, time_i, time_01):

    d_times = time_01[1] - time_01[0]
    d_ti = time_i - time_01[0]
    s = d_ti / d_times
    # p01 = np.matmul(p0, np.linalg.inv(p1))

    rot0 = p0[0:3, 0:3]
    rot1 = p1[0:3, 0:3]

    rots_obj = scipyR.from_matrix([rot0, rot1])
    slerp = Slerp(time_01, rots_obj)
    times = [time_01[0], time_i, time_01[1]]
    interp_rots = slerp(times)

    t0i = (1-s) * p0[0:3, 3] + s * p1[0:3, 3]
    p0i = np.identity(4, dtype=np.float32)
    p0i[0:3, 0:3] = interp_rots.as_matrix()[1]
    p0i[0:3, 3] = t0i

    # np.matmul(p0, p0i)
    return p0i


def t_pose_interp():
    # Test1: plot poses and interpolated pose
    p0 = get_random_pose(None, [-0.23, 0.55, 0.12])
    p1 = get_random_pose(None, [0.13, 0.055, -0.007])

    pi = interp_pose(p0, p1, 0.1, [0.012, 1.108])
    pi0 = interp_pose(p0, p1, 0.22, [0.012, 1.108])
    pi1 = interp_pose(p0, p1, 0.35, [0.012, 1.108])
    pi2 = interp_pose(p0, p1, 0.5, [0.012, 1.108])
    pi3 = interp_pose(p0, p1, 0.75, [0.012, 1.108])
    pi4 = interp_pose(p0, p1, 0.85, [0.012, 1.108])
    pi5 = interp_pose(p0, p1, 0.94, [0.012, 1.108])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Setting axis to equal is a disaster in matplotlib!
    # ax.set_aspect('equal')
    # axisEqual3D(ax)

    mviz.draw_coord_sys(ax, p0, 0.05, 'O0')
    mviz.draw_coord_sys(ax, pi, 0.05, 'Oi')
    mviz.draw_coord_sys(ax, pi0, 0.05, 'Oi0')
    mviz.draw_coord_sys(ax, pi1, 0.05, 'Oi1')
    mviz.draw_coord_sys(ax, pi2, 0.05, 'Oi2')
    mviz.draw_coord_sys(ax, pi3, 0.05, 'Oi3')
    mviz.draw_coord_sys(ax, pi4, 0.05, 'Oi4')
    mviz.draw_coord_sys(ax, pi5, 0.05, 'Oi5')
    mviz.draw_coord_sys(ax, p1, 0.05, 'O1')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    t_pose_interp()


