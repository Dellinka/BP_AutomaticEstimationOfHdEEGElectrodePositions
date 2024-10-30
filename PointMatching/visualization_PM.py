import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


def visualize_epipolar_lines(img1, img2, F, p1, p2=None, save_f=""):
    """
    This function plots two figure with epipolar line in img2 from points p1 and fundamental matrix F,
    points p2 will be also plot in the second image. This function is used for visual check of
    possibly predicted points.

    :param save_f:          Optional parameter for saving image - path to file
                            File should be named as '<subject>_cam1_cam2_cam3.png'
    :param img1:            Path to first image with camera matrix P1
    :param img2:            Path to second image with camera matrix P2
    :param p1:              Points of 2d coordinates in image1 as np.array (2, N)
    :param p2:              Points of 2d coordinates in image2 as np.array (2, M)
    :param F:               Fundamental matrix showing relationship between first and second camera
    """
    def to_homogeneous(p):
        """Return homogeneous coordinates from given point"""
        N = p.shape[1]
        p = np.vstack((p, np.ones(N)))
        return p

    def plot_lines_and_points(ax, image_path, lines=None, points=None):
        """Plot given points and line between them"""
        img = cv2.imread(image_path)
        h, w, _ = np.shape(img)
        ax.set_ylim([h, 0])
        ax.set_xlim([0, w])
        ax.imshow(img)

        if lines is not None:
            for line in np.transpose(lines):
                s, t, u = line[0], line[1], line[2]
                x = np.linspace(0, w, 100)
                y = (0 - s * x - u) / t
                ax.plot(x, y, '-r', linewidth=1)

        if points is not None:
            for point in np.transpose(points):
                point = patches.Circle(point, 3, color='red')
                ax.add_patch(point)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_dpi(300)

    plot_lines_and_points(ax2, img2, lines=F @ to_homogeneous(p1))
    plot_lines_and_points(ax1, img1, points=to_homogeneous(p1))

    if p2 is not None:
        plot_lines_and_points(ax2, img2, points=to_homogeneous(p2))

    fig.suptitle("Epipolar lines computed from fundamental matrix")
    ax1.set_title("Point from {}".format(os.path.basename(img1)))
    ax2.set_title("Epipolar line in {}".format(os.path.basename(img2)))

    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()
