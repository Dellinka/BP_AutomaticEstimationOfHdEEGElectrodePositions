"""
This script evaluates computed camera projection matrix by computation of average reprojection error
between original and reprojected points using specified camera matrix.
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from CameraComputation import compute_reprojected_points, compute_camera_matrix, create_average_matrices
from reader import load_points


def compute_average_reprojection_error(x, X, P):
    """
    This function computes average reprojection error between image points and points
    reprojected from 3D scene points and camera matrix P

    :param x        2D image points             ~ np.array(2, N) where N is number of points
    :param X        3D world scene points       ~ np.array(3, N) where N is number of points
    :param P:       Camera projection matrix P  ~ np.array(3, 4)

    :return:        Reprojection error          ~ float
    """
    # Compute reprojected points
    x_proj = compute_reprojected_points(X, P)
    x_orig = np.transpose(x)

    # Compute the reprojection error
    sum = 0
    _, points_num = np.shape(X)
    for i in range(0, points_num):
        sum += np.sqrt(np.square(x_orig[i][0] - x_proj[i][0]) + np.square(x_orig[i][1] - x_proj[i][1]))
    err = sum/points_num

    return err


def evaluate_subject(subject, dir_2d, dir_3d, camera_matrices_dir):
    """
    This function evaluates computed average camera matrices. Compute
    average reprojection error between original and reprojected points using the computed camera matrix
    for subject and average camera matrix between all subjects.

    :param subject                  Name of the subject to be evaluated
    :param dir_2d                   Path to directory 2d_annotations_csv_orig
                                    (inside are all subjects and their 2d annotation in csv file (Sensor id | x | y))
    :param dir_3d                   Path to directory 3d_annotations_csv_orig
                                    (inside are all subjects and their 3d annotation in csv file (Sensor id | x | y | z))
    :param camera_matrices_dir      Path to directory with computed camera matrices for all cameras

    """
    errs_P = {}
    err_avg_P = {}
    for cam_num in range(1, 12):
        name = 'cam' + str(cam_num)

        file_2d = os.path.join(dir_2d, subject, 'camera' + str(cam_num) + '.csv')
        file_3d = os.path.join(dir_3d, subject + '.csv')
        pts_2d, pts_3d = load_points(file_2d, file_3d)
        P = compute_camera_matrix(pts_2d, pts_3d)
        avg_P = np.loadtxt(os.path.join(camera_matrices_dir, 'camera' + str(cam_num) + '.out'))

        errs_P[name] = (round(compute_average_reprojection_error(pts_2d, pts_3d, P), 4))
        err_avg_P[name] = (round(compute_average_reprojection_error(pts_2d, pts_3d, avg_P), 4))

    return errs_P, err_avg_P


def visualize_reprojection(x, X, P, img_file, save_f=""):
    """
    This function plots matplotlib figure with original and projected points from camera matrix P.

    :param x            2D image points             ~ np.array(2, N) where N is number of points
    :param X            3D world scene points       ~ np.array(3, N) where N is number of points
    :param P:           Camera projection matrix P  ~ np.array(3, 4)
    :param img_file:    Path to image
    """
    x_proj = compute_reprojected_points(X, P)
    x_orig = np.transpose(x)

    fig, ax = plt.subplots(1)
    fig.set_dpi(300)
    fig.suptitle('Reprojection error = {}'.format(compute_average_reprojection_error(x, X, P)))

    img = cv2.imread(img_file)
    xorg = x_orig[:, 0]
    yorg = x_orig[:, 1]
    xproj = x_proj[:, 0]
    yproj = x_proj[:, 1]

    color_a = "deepskyblue"
    color_b = "tomato"

    for xx, yy in zip(xorg, yorg):
        circ = patches.Circle((xx, yy), 4, color=color_a)
        ax.add_patch(circ)

    for xx, yy in zip(xproj, yproj):
        circ = patches.Circle((xx, yy), 4, color=color_b)
        ax.add_patch(circ)

    legend = [patches.Patch(color=color_a, label='Original points'),
              patches.Patch(color=color_b, label='Reprojected points')]
    ax.legend(handles=legend, fontsize=10)

    ax.imshow(img)
    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))
        plt.savefig(save_f)
    else:
        plt.show()


def cross_validation(subjects_all, dir_2d, dir_3d, output_dir='cameras_evaluation'):
    """
    Leave one out cross validation - camera matrices are computed from all except one subject,
    then the subject is used for validation. This process is run through all the subjects.

    :param subjects_all:            Array of subjects fom which to evaluate
    :param dir_2d:                  Directory to 2D original data
    :param dir_3d:                  Directory to 3D original data
    :return:
    """
    err_P = {'cam1': [], 'cam2': [], 'cam3': [], 'cam4': [], 'cam5': [], 'cam6': [], 'cam7': [], 'cam8': [], 'cam9': [], 'cam10': [], 'cam11': []}
    err_avg_P = {'cam1': [], 'cam2': [], 'cam3': [], 'cam4': [], 'cam5': [], 'cam6': [], 'cam7': [], 'cam8': [], 'cam9': [], 'cam10': [], 'cam11': []}
    for _ in range(len(subjects_all)):
        subject = subjects_all[0]
        subjects_all.pop(0)
        print("Average camera matrices computed from subjects ", str(subjects_all))
        create_average_matrices(subjects_all, dir_2d, dir_3d, output_dir)
        print("Evaluation on subject ", subject)
        err_P_computed, err_avg_P_computed = evaluate_subject(subject, dir_2d, dir_3d, output_dir)
        subjects_all.append(subject)

        for key, value in err_avg_P_computed.items():
            err_avg_P[key].append(value)

        for key, value in err_P_computed.items():
            err_P[key].append(value)

    for key, value in err_avg_P.items():
        print("-----------")
        print("{} with average reprojected error {}, min {} and max {}".
              format(key, np.average(np.asarray(value)), np.min(np.asarray(value)), np.max(np.asarray(value))))
        print(value)


if __name__ == '__main__':
    subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JK", "JR", "LK",
                "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "TN", "VV"]

    input_dir_2d = os.path.join('..', 'Dataset', '2d_annotations_csv_orig')
    input_dir_3d = os.path.join('..', 'Dataset', '3d_annotations_csv_orig')

    cross_validation(subjects, input_dir_2d, input_dir_3d)
