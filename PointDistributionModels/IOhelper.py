"""
This script contains help functions for input and output such as reading and saving data.
"""
import csv
import os

import numpy as np


def read_points(file):
    """
    This function reads 3D coordinates from input csv file. Each line represents one point
    where first column is the name of sensor and other three are x, y and z coordinates.

    :param file:            Path to csv file with 3D coordinates
    :return: points         Vector with read points as np.array(3N, ) where N is number of points
             labels         Array of labels of points corresponding with points - list of size N

    """
    with open(file) as file_content:
        csv_reader = csv.reader(file_content, delimiter=';')
        labels = list()
        points = list()

        for row in csv_reader:
            labels.append(row[0])
            points.append(float(row[1]))
            points.append(float(row[2]))
            points.append(float(row[3]))

    return np.asarray(points), np.asarray(labels)


def read_3Dcoords(file):
    """
    This function reads 3d coordinates from given file

    :param file:            Path to file with 3d coordinates
    :return:                Points and corresponding labels as numpy array
    """
    with open(file) as file_content:
        lines = file_content.read().splitlines()
        labels = list()
        points = list()

        for row in lines:
            line = row.split()
            labels.append(line[0])
            points.append(float(line[1]))
            points.append(float(line[2]))
            points.append(float(line[3]))

    return np.asarray(points), np.asarray(labels)


def read_predicted_model(file):
    """
    This function reads 3D coordinates from input csv file (the generated correspondences).
    Each line represents one correspondence where we only look at first column containing 3D coordinates
    separated with space.
    Note: The first row of input file is title - should be skipped.

    :param file:            Path to csv file with 3D coordinates
    :return: points         Vector with read points as np.array(3N, ) where N is number of points
             colors         Array of predicted sensor colors (0 - black, 1 - white) corresponding with points np.array(N, )
    """
    with open(file) as file_content:
        csv_reader = csv.reader(file_content, delimiter=';')
        points = list()
        colors = list()

        first = True
        for row in csv_reader:
            if first:
                first = False
                continue

            [points.append(float(p)) for p in row[0].split()]
            colors.append(int(row[1]))

    points = np.asarray(points).reshape((int(len(points) / 3), 3))
    return points, np.asarray(colors)


def save_statistical_model(model, labels, output_dir, assign_matrix=None, cov_matrix=None):
    """
    This function saves computed statistical model into config directory.

    :param model:                   Model shape (mean of models for statistical model or transformed mean) np.array(3N, )
    :param labels:                  Names of electrodes corresponding with mean points
    :param cov_matrix:              Covariance matrix np.array(3N, 3N)
    :param output_dir:              Directory for output

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savetxt(os.path.join(output_dir, 'model.out'), model)
    np.savetxt(os.path.join(output_dir, 'model_labels.out'), labels, fmt="%s")
    if assign_matrix is not None:
        np.savetxt(os.path.join(output_dir, 'assign_matrix.out'), assign_matrix)
    if cov_matrix is not None:
        np.savetxt(os.path.join(output_dir, 'covariance_matrix.out'), cov_matrix)


def save_computed_coordinates(labels, coordinates, output_dir):
    """
    This function saves computed 3D coordinates in generated data directory.

    :param labels:              Array of labels so saved coordinates are ordered (260, )
    :param coordinates:         Dictionary as {label: np.array(3,)}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = open(os.path.join(output_dir, '3d_model.out'), 'w')
    for label in labels:
        coord = coordinates[label]
        f.write(label + '\t\t' +
                str(round(coord[0], 5)) + '      ' +
                str(round(coord[1], 5)) + '      ' +
                str(round(coord[2], 5)) + '\n')
