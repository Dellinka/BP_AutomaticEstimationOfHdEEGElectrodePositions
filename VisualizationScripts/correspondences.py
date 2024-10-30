"""
This script visualizes found correspondences between images via matplotlib library. Different types
of visualizations can be choose - read bellow.

visualize_corresponding_points          Visualize correspondences specified in to_plot array
                                        (each correspondence in separated plot)
visualize_all_correspondences           Visualize all correspondences in all images in one plot with two
                                        different colors for different number of corresponding points
visualize_all_different_pts             Visualize all correspondences from two different datasets of points -
                                        CORRESPONDING_DATA2 is used in this function

visualize_3_correspondences_line        Visualize all correspondences between three images with lines

visualize_2_correspondences_line        Visualize correspondences between two images with lines


"""
import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from random import randint, shuffle

subject = 'TD'
IMAGE_DIR = os.path.join('..', 'Dataset', 'images_dir', subject)
CORRESPONDING_DATA = os.path.join('..', 'GeneratedData', 'Correspondences', subject, 'corresponding_points.csv')
CORRESPONDING_DATA2 = os.path.join('..', 'GeneratedData', 'Correspondences', subject, 'corresponding_points.csv')

# IMAGE_DIR = os.path.join('..', 'RESULTS', 'images')
# CORRESPONDING_DATA = os.path.join('..', 'RESULTS', 'Correspondences', 'corresponding_points.csv')
SAVE_DIR = ''   # os.path.join('..', 'Images', 'Corresponding', subject + '_3+_eps5_8.png')

# used in visualize_all_different_pts()
# CORRESPONDING_DATA2 = os.path.join('..', 'RESULTS', 'Correspondences', 'corresponding_points.csv')
# Used in visualize_corresponding_points()
TO_PLOT = [i for i in range(0, 15)]
# Used in visualize_correspondences_line
cam1 = 2
cam2 = 7
cam3 = 9


class CorrespondingPoints:
    """
    Wrapper for consistent / possibly corresponding points. These points have the same color
    and their projections from 3D point are consistent (differ in less than 10 px).

    Note:  self.points is a set of possibly corresponding points as tuples (documentation => set 'C')
           self.points =  {camera: (x, y),                    Corresponding points through different images
                           ... })
    """

    def __init__(self, sensor_color):
        self.color = sensor_color
        self.coord_3d = np.ones(3)
        self.points = {}
        self.points_reprojected = {}


def read_correspondences(input_file):
    """
    Read correspondences from csv file.

    :param input_file:              Path to input csv file with correspondences

    :return correspondences         Array of CorrespondingPoints
    """
    correspondences = list()

    with open(input_file) as file_content:
        csv_reader = csv.reader(file_content, delimiter=';')
        index = 0

        for row in csv_reader:
            if index == 0:
                index += 1
                continue

            points = CorrespondingPoints(int(row[1]))
            coord_3s_str = row[0].split()
            points.coord_3d = np.array([float(coord_3s_str[0]),
                                        float(coord_3s_str[1]),
                                        float(coord_3s_str[2])])

            for idx, cell in enumerate(row):
                if idx == 0 or idx == 1: continue
                if idx % 2 == 1 or len(cell.strip()) == 0: continue
                camera = int(cell)
                coord_2d_str = row[idx+1].split()
                points.points[camera] = (float(coord_2d_str[0]), float(coord_2d_str[1]))

            correspondences.append(points)

    return correspondences


def visualize_corresponding_points(image_dir, corresponding_pts, to_plot=None, save_f=""):
    """
    This function plots correspondence specified in to_plot array.

    :param to_plot:                         Optional array of points to plot
    :param image_dir:                       Path to directory with images
    :param corresponding_pts:               Corresponding points as array of CorrespondingPoints
    """

    def plot_point_help(ax, img, p, color='red'):
        img = cv2.imread(img)
        point = patches.Circle([round(p[0]), round(p[1])], 4, color=color)
        ax.imshow(img)
        ax.add_patch(point)

    if to_plot is None:
        to_plot = [randint(0, len(corresponding_pts) - 1) for _ in range(10)]
    corresponding_points_array = [corresponding_pts[idx] for idx in to_plot]

    for i, corresponding_points in enumerate(corresponding_points_array):
        corresponding_points = corresponding_points.points
        fig = plt.figure()
        fig.set_dpi(300)
        plt.suptitle("Corresponding point with id " + str(to_plot[i]))
        N = len(corresponding_points)
        rows = int(np.ceil(N / 2))

        gs = gridspec.GridSpec(rows, 2)
        gs.update(wspace=0.5, hspace=0.5)

        idx = 0
        for camera, point in corresponding_points.items():
            ax_tmp = fig.add_subplot(gs[int(np.floor(idx / 2)), idx % 2])
            plot_point_help(ax_tmp, os.path.join(image_dir, 'camera' + str(camera) + '.png'), point)

            ax_tmp.set_title("Camera {}".format(camera))
            idx += 1

        if len(save_f) != 0:
            if not os.path.exists(save_f):
                os.makedirs(save_f)

            plt.savefig(os.path.join(save_f, 'pt_' + str(to_plot[i]) + '.png'))
        else:
            plt.show()


def visualize_all_correspondences(image_dir, corresponding_pts, save_f=""):
    """
    This function visualize all 11 images and predicted correspondences in different color according to number
    of corresponding points. Used for visualization of problem with Gurobi preference correspondence finding method.
    
    :param image_dir:                       Path to directory with images
    :param corresponding_pts:               Corresponding points as array of CorrespondingPoints
    :param save_f: (optional)               Save image in this file
    """
    # Prepare image axes and grid
    fig = plt.figure()
    fig.set_dpi(300)
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.1, hspace=0.1)

    # Plot images
    ax_arr = list()
    for c in range(0, 11):
        ax_arr.append(fig.add_subplot(gs[c // 4, c % 4]))
        ax_arr[c].axes.xaxis.set_visible(False)
        ax_arr[c].axes.yaxis.set_visible(False)
        img = cv2.imread(os.path.join(image_dir, 'camera' + str(c+1) + '.png'))
        ax_arr[c].imshow(img)

    # Plot points
    for idx, corresponding in enumerate(corresponding_pts):
        color = 'r' if len(corresponding.points) > 2 else 'deepskyblue'
        for camera, p in corresponding.points.items():
            p = patches.Circle([round(p[0]), round(p[1])], 4, color=color)
            ax_arr[camera-1].add_patch(p)

    plt.suptitle('Correspondences with two (blue) and more (red) points')
    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()


def visualize_all_different_pts(image_dir, corresponding_points, corresponding_points2, save_f=""):
    """
    Visualize all correspondences from two different datasets of points. Also used for visualization of problem
    with Gurobi preference correspondence finding method (correspondences with only two and only three points).

    :param image_dir:                       Path to directory with images
    :param corresponding_points:            Corresponding points as array of CorrespondingPoints
    :param corresponding_points2:           Second dataset with corresponding points as array of CorrespondingPoints
    :param save_f: (optional)               Save image in this file
    """
    # Prepare image axes and grid
    fig = plt.figure()
    fig.set_dpi(300)
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.1, hspace=0.1)

    # Plot images
    ax_arr = list()
    for c in range(0, 11):
        ax_arr.append(fig.add_subplot(gs[c // 4, c % 4]))
        ax_arr[c].axes.xaxis.set_visible(False)
        ax_arr[c].axes.yaxis.set_visible(False)
        img = cv2.imread(os.path.join(image_dir, 'camera' + str(c+1) + '.png'))
        ax_arr[c].imshow(img)

    # Plot points
    for idx, corresponding in enumerate(corresponding_points):
        color = 'b'
        for camera, p in corresponding.points.items():
            p = patches.Circle([round(p[0]), round(p[1])], 10, color=color)
            ax_arr[camera-1].add_patch(p)

    for idx, corresponding in enumerate(corresponding_points2):
        color = 'r'
        for camera, p in corresponding.points.items():
            p = patches.Circle([round(p[0]), round(p[1])], 5, color=color)
            ax_arr[camera-1].add_patch(p)

    plt.suptitle('All corresponding points from two different datasets')
    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()


def visualize_3_correspondences_line(image_dir, corresponding_points, cameras_num, number=5, save_f=""):
    """
    This function three point correspondences between given cameras as points in images and lines between them.

    :param image_dir:                       Path to directory with images
    :param corresponding_points:            Corresponding points as array of CorrespondingPoints
    :param cameras_num:                     Array of 3 camera numbers
    :param number:                          Maximal number of  visualized correspondences
    :param save_f: (optional)               Save image in this file
    """
    def plot_img(ax, camera):
        img = cv2.imread(os.path.join(image_dir, 'camera' + str(camera) + '.png'))
        ax.imshow(img)

    def plot_line_three(axs, points, color, id):
        for index, pt in enumerate(points):
            axs[index].plot(pt[0], pt[1], 'o', markersize=1.5, color=color)
            axs[index].text(pt[0], pt[1], str(id), fontsize='6', color=color)

            con = patches.ConnectionPatch(xyA=points[index], xyB=points[(index+1) % 3],
                                          coordsA="data", coordsB="data",
                                          axesA=axs[index], axesB=axs[(index+1) % 3], color=color)
            con.set_linewidth(0.8)
            axs[(index+1) % 3].add_artist(con)

            con = patches.ConnectionPatch(xyA=points[index], xyB=points[(index + 1) % 3],
                                          coordsA="data", coordsB="data",
                                          axesA=axs[index], axesB=axs[(index + 1) % 3], color=color)
            con.set_linewidth(0.8)
            axs[index].add_artist(con)

    if len(cameras_num) != 3:
        print("ERROR: specify the number of cameras as array of size 3")
        return None

    # Read colors
    f = open('colors.txt', "r")
    colors = [line[:-1] for line in f]
    f.close()


    # Prepare image axes and grid
    fig = plt.figure()
    fig.set_dpi(300)
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 1:3])
    axis = [ax1, ax2, ax3]

    for idx, cam in enumerate(cameras_num):
        plot_img(axis[idx], cam)

    # Plot points and lines between correspondences
    idx = 0
    idx_c = 0
    for correspondences in corresponding_points:
        points = list()
        for cam in cameras_num:
            points.append(correspondences.points.get(cam))

        if sum(x is None for x in points) == 0:
            color = colors[idx].strip()
            plot_line_three(axis, points, color, idx_c)
            idx_c += 1
            idx += 1

        if idx >= len(colors):
            shuffle(colors)
            idx = 0

    # plt.suptitle("Correspondences between images from camera {} and {}".format(1, 2, 3))
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)

    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()

    return None


def visualize_2_correspondences_line(image_dir, corresponding_points, camera1, camera2, save_f=""):
    """
    This function visualize correspondences as points in two specified images and lines between them.

    :param image_dir:                       Path to directory with images
    :param corresponding_points:            Corresponding points as array of CorrespondingPoints
    :param camera1                          Number of first camera to visualize
    :param camera2                          Number of second camera to visualize
    :param save_f: (optional)               Save image in this file
    """
    # Read colors
    f = open('colors.txt', "r")
    colors = [line[:-1] for line in f]
    f.close()

    # Prepare image axes and grid
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    fig.set_dpi(300)

    # Plot images
    img = cv2.imread(os.path.join(image_dir, 'camera' + str(camera1) + '.png'))
    ax1.set_title("Camera {}".format(camera1))
    ax1.imshow(img)

    img = cv2.imread(os.path.join(image_dir, 'camera' + str(camera2) + '.png'))
    ax2.set_title("Camera {}".format(camera2))
    ax2.imshow(img)

    # Plot points and lines between correspondences
    idx = 0
    idx_c = 0
    for correspondences in corresponding_points:
        point_cam1 = correspondences.points.get(camera1)
        point_cam2 = correspondences.points.get(camera2)
        if point_cam1 is None or point_cam2 is None:
            continue

        if idx >= len(colors):
            shuffle(colors)
            idx = 0
        color = colors[idx].strip()

        ax1.plot(point_cam1[0], point_cam1[1], 'o', markersize=1.5, color=color)
        ax1.text(point_cam1[0], point_cam1[1], str(idx_c), fontsize='6', color=color)
        ax2.plot(point_cam2[0], point_cam2[1], 'o', markersize=1.5, color=color)
        ax2.text(point_cam2[0], point_cam2[1], str(idx_c), fontsize='6', color=color)
        idx_c += 1
        idx += 1

        con = patches.ConnectionPatch(xyA=point_cam1, xyB=point_cam2, coordsA="data", coordsB="data",
                                      axesA=ax1, axesB=ax2, color=color)
        con.set_linewidth(0.8)
        ax2.add_artist(con)

    plt.suptitle("Correspondences between images from camera {} and {}".format(camera1, camera2))
    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()


if __name__ == '__main__':
    # visualize_corresponding_points(IMAGE_DIR, read_correspondences(CORRESPONDING_DATA), to_plot=TO_PLOT, save_f=SAVE_DIR)
    # visualize_all_correspondences(IMAGE_DIR, read_correspondences(CORRESPONDING_DATA), save_f=SAVE_DIR)
    # visualize_all_different_pts(IMAGE_DIR, read_correspondences(CORRESPONDING_DATA), read_correspondences(CORRESPONDING_DATA2), save_f=SAVE_DIR)
    # visualize_3_correspondences_line(IMAGE_DIR, read_correspondences(CORRESPONDING_DATA), [cam1, cam2, cam3], save_f="")
    visualize_2_correspondences_line(IMAGE_DIR, read_correspondences(CORRESPONDING_DATA), cam1, cam2, save_f=SAVE_DIR)
