import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_nn_elements(nn_elems):
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # import matplotlib.pylab as pylab

    idxs = np.array([i for i in range(0, len(nn_elems))])
    e1 = np.array([el1 for el1, el2, el3 in nn_elems])
    e2 = np.array([el2 for el1, el2, el3 in nn_elems])
    e3 = np.array([el3 for el1, el2, el3 in nn_elems])

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(idxs, e1, '-', color="blue")
    ax.plot(idxs, e2, '-.', color="red")
    ax.plot(idxs, e3, '--', color="green")
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('translational error [m]')
    plt.show()

    return 1


def draw_box_stats(ax, stats, pos_x, box_label, box_color='blue', box_style='-', box_width=0.5):

    box_hw = box_width / 2
    box_qw = box_hw / 2

    first_h, = ax.plot([pos_x - box_qw, pos_x + box_qw], [stats[0], stats[0]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[1], stats[1]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[2], stats[2]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[3], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_qw, pos_x + box_qw], [stats[-1], stats[-1]], label=box_label, color=box_color, linestyle=box_style)

    ax.plot([pos_x, pos_x], [stats[0], stats[1]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x - box_hw], [stats[1], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x + box_hw, pos_x + box_hw], [stats[1], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x, pos_x], [stats[3], stats[-1]], label=box_label, color=box_color, linestyle=box_style)

    return first_h


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_kmeans_pp(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='next centroid')
    plt.title('Select % d th centroid' % (centroids.shape[0]))

    plt.legend()
    plt.xlim(-5, 12)
    plt.ylim(-10, 15)
    plt.show()


def show_color_segments(frame, counts_map, n_dominant_colors, close_ksize):
    sys.path.append('../')
    import cv_mine.cv_utils as cvu

    cnt2 = 0
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (0, 255, 255), (255, 255, 0), (255, 0, 255),
              (255, 255, 255), (255, 127, 0), (127, 0, 255), (0, 127, 255)]
    b, g, r = cv2.split(frame)
    descending_counts = sorted(counts_map.keys(), reverse=True)

    for count in descending_counts:
        color_voxels = counts_map[count]

        for color_voxel in color_voxels:
            bin_image = np.uint8(cvu.color_thresh(b, g, r, color_voxel['group'])) * 255

            # fill the holes with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ksize)
            res = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)

            frame[res > 0] = colors[cnt2]

        cnt2 += 1
        if cnt2 >= n_dominant_colors:
            break

    return frame


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # draw_coord_sys(ax, np.identity(4, dtype=np.float32), 0.05, 'O')
    # plot_traj_3d(ax, np.array([[0, 0, 0], [1, 2, -1], [2, 1, 0]]), 'bili', ['c', '*', 'r'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
