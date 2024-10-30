"""
In this file are functions used for visualization of models used in different scripts.
"""

import os

import matplotlib.pyplot as plt

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


def visualize_models(points, labels=None, title="", save_f=""):
    """
    This function draw 3D points via matplotlib library.

    :param points:          Array of 3D points in numpy vector to be visualize... [np.array(3N, )]
    :param labels:          Array of labels corresponding to points
    :param save_f:          Optional parameter for saving image - path to file
    :param title:           Optional parameter for plot title
    """
    if labels is not None and len(labels) != len(points):
        print("Different number of models and labels")

    ax = plt.axes(projection='3d')
    plt.gcf().set_dpi(300)

    for idx, pts in enumerate(points):
        pts = pts.reshape((int(len(pts) / 3), 3))
        if labels is not None:
            ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], label=labels[idx])
        else:
            ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])
        # ax.view_init(30, 60)
        ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-13, 13)
    ax.set_ylim3d(-13, 13)
    ax.set_zlim3d(-13, 13)
    ax.set_title(title)

    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()


def visualize_correspondences(pts_x, pts_y, labels=None, title="", save_f=""):
    """
    This function visualize given models with line between corresponding points.

    :param pts_x:               First 3D Model points corresponding to pts_y
    :param pts_y:               Second 3D Model points corresponding to pts_x
    :param labels:              Labels of models as array of size 2
    :param title: (optional)    Title for plotted image
    :param save_f: (optional)   Save image in this file
    """
    if labels is not None and len(labels) != 2:
        print("Different number of models and labels")

    pts_x = pts_x.reshape((int(len(pts_x) / 3), 3))
    pts_y = pts_y.reshape((int(len(pts_y) / 3), 3))

    ax = plt.axes(projection='3d')
    plt.gcf().set_dpi(300)

    for idx in range(len(pts_x)):
        x = pts_x[idx]
        y = pts_y[idx]
        ax.plot3D([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], 'r-')

    if labels is not None:
        ax.scatter3D(pts_x[:, 0], pts_x[:, 1], pts_x[:, 2], label=labels[0])
        ax.scatter3D(pts_y[:, 0], pts_y[:, 1], pts_y[:, 2], label=labels[1])
    else:
        ax.scatter3D(pts_x[:, 0], pts_x[:, 1], pts_x[:, 2])
        ax.scatter3D(pts_y[:, 0], pts_y[:, 1], pts_y[:, 2])

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-13, 13)
    ax.set_ylim3d(-13, 13)
    ax.set_zlim3d(-13, 13)
    ax.set_title(title)

    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()
