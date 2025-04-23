import numpy as np
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp


def calc_dist_identity_quat(curr_trans):
    """
    Calculates the distance of pose represented as [x y z qx qy qz qw] to identity transform
    :param curr_trans:
    :return:
    """
    return np.sum(np.abs(curr_trans - [0, 0, 0, 0, 0, 0, 1]))


def is_identity_quat(curr_trans, th=1e-6):
    """
    :param curr_trans: input transformation: [tx ty tz qx qy qz qw]
    :param th:
    :return: True: is identity transform (I4x4)
            False: is not identity
    """
    return calc_dist_identity_quat(curr_trans) < th


def swap_qw(pose_list):
    """
    Swaps qw in a list of poses like: l[ts x y z qw qx qy qz] to l[... z qx qy qz qw]
    :param pose_list:
    :return:
    """
    for gt in pose_list:
        sval = gt[4]
        gt[4:7] = gt[5:]
        gt[7] = sval

    return pose_list


def break_pose_graph(pose_list, th_ts=1e-12, th_iden=1e-3):
    """
    Breaks pose chain by equal timestamps or identity transform
    :param pose_list: [[ts0 x0 y0 z0 qx0 qy0 qz0 qw0] ...]
    :param th_ts:
    :param th_iden:
    :return: list of consecutive poses [[[x0 y0 ...],[x1 y1 ...]...]...]
    """
    posePieces = []
    nList = len(pose_list)

    if nList < 3:
        posePieces.append(pose_list)
        return posePieces

    lastPose = pose_list[0]
    currPiece = []
    currPiece.append(lastPose)

    for i in range(1, nList):
        currPose = pose_list[i]
        if abs(currPose[0]-lastPose[0]) < th_ts or is_identity_quat(currPose[1:], th_iden):

            posePieces.append(currPiece)
            lastPose = currPose
            currPiece = []
            currPiece.append(lastPose)
        else:
            currPiece.append(currPose)
            if i == nList-1:
                posePieces.append(currPiece)

    return posePieces


def select_best_pose_piece(posePieces):

    if len(posePieces) <= 0:
        return []

    max_len = 0
    max_pg = []

    for posePiece in posePieces:
        if max_len < len(posePiece):
            max_len = len(posePiece)
            max_pg = posePiece

    return max_pg


def list2dict(pose_list):
    """
    Convert a list of floats: [ts x y ...] to dict: [ts, [x y ...]]
    :param pose_list:
    :return:
    """
    bin_list = [[pose[0], pose[1:]] for pose in pose_list]
    return dict(bin_list)


def sort_pose_list(pose_list):
    pose_dict = list2dict(pose_list)
    pose_ts = list(pose_dict.keys())
    pose_ts.sort()
    sorted_pose = []
    for ts in pose_ts:
        curr_pose = []
        curr_pose.append(ts)
        for val in pose_dict[ts]:
            curr_pose.append(val)
        sorted_pose.append(curr_pose)
    return sorted_pose


def get_stats_from_data(data):
    """
    :param data: numeric list
    return: [min, q1, q2, q3, max] of data
    """

    data_sorted = np.sort(data)
    n_data = len(data)
    stats = np.array([data_sorted[0], data_sorted[int(n_data * 0.25)], data_sorted[int(n_data * 0.5)],
                      data_sorted[int(n_data * 0.75)], data_sorted[-1]])

    return stats


def get_stats_errs_file(file_name, trans_info, rot_info):

    with open(file_name, 'r') as stats_file:

        data = stats_file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")

        method_label = ''
        curr_info = dict()

        n_lines = len(lines)

        for i in range(0, n_lines):
            line = lines[i].strip()
            if len(line) <= 0:
                continue
            if line[0] == '#':
                if ':' in line:
                    header_parts = line[1:].split(':')
                    method_label = header_parts[0].strip()

                    if 'trans' in header_parts[1].strip():
                        trans_info[method_label] = dict()
                        curr_info = trans_info[method_label]

                    elif 'rot' in header_parts[1].strip():
                        rot_info[method_label] = dict()
                        curr_info = rot_info[method_label]

            else:
                curr_stat_list = [np.float64(v) for v in line.split(' ')]
                curr_info[curr_stat_list[0]] = curr_stat_list[1:]

    return trans_info, rot_info


def find_intersection(info_obj):

    res_list = []
    for method_key in list(info_obj.keys()):

        curr_method = info_obj[method_key]

        if len(res_list) <= 0:
            res_list = set(curr_method.keys())
        else:
            res_list &= set(curr_method.keys())

    return list(res_list)


def perform_test1():

    # Creating dataset
    np.random.seed(10)

    data = np.random.normal(100, 20, 200)
    data1 = np.random.normal(100, 20, 200)

    stats = get_stats_from_data(data)
    stats1 = get_stats_from_data(data1)

    pos_x = 1

    fig = plt.figure(figsize=(10, 7))

    # Creating plot
    # plt.boxplot(stats)

    box_color = 'blue'
    box_style = '--'

    mviz.draw_box_stats(stats, pos_x, box_color, box_style)
    mviz.draw_box_stats(stats1, pos_x + 0.6, 'red', '-')

    # show plot
    plt.show()


if __name__ == "__main__":

    key_rots = scipyR.random(2, random_state=2342345)
    p0 = np.identity(4)
    p0[0:3, 0:3] = key_rots.as_matrix()[0]
    p0[0:3, 3] = np.array([-0.23, 0.55, 0.12]).transpose()
    p1 = np.identity(4)
    p1[0:3, 0:3] = key_rots.as_matrix()[1]
    p1[0:3, 3] = np.array([0.13, 0.055, -0.007]).transpose()

    pi = interp_pose(p0, p1, 0.35, [0.012, 1.108])

    print("Testing utils:")
    trans = np.array([-1.23448e-07, -1.33685e-07, -3.18998e-07, -2.8187e-05, -4.26486e-05, 8.75161e-05, 1])
    print("Transform: ")
    print(trans)
    if is_identity_quat(trans, 1e-3):
        print("Is identity")
    else:
        print("Is not identity")
    trans = np.array([-0.0823314, -0.0130145, -0.0204296, -0.00542771, -0.0044356, 0.00535609, 0.999961])
