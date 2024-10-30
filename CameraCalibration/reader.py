"""
This script is used for reading 2d and 3d points as numpy array, which can be then used for camera matrix computation.
"""
import csv
import numpy as np


def load_points(file_2d, file_3d):
    """
    This function reads csv file and returns read 2d image points and 3d scene points.

    :param file_2d:                 Path to file with 2D image points
    :param file_3d:                 Path to file with 3D scene points
    :return: points_2d, points_3d   Numpy array with points 2D and 3D points np.array(2, N) and np.array(3, N)
    """
    indexes = list()
    points_2d = list()

    # Load the 2d points coordinate first and save their sensors indexes in idx list
    first_line = True
    with open(file_2d) as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            if first_line:
                first_line = False
                continue

            indexes.append(int(row[0]))
            points_2d.append([int(float(row[1])),
                              int(float(row[2]))])

    points_2d = np.transpose(np.asarray(points_2d))

    # Load 3d points coordinates and save them in dictionary
    dict_3d = {}
    with open(file_3d) as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            dict_3d[row[0]] = np.array([float(row[1]), float(row[2]), float(row[3])])

    # Create array with 3d points corresponding to points_2d
    points_3d = list()
    for idx in indexes:
        if idx == 257:
            points_3d.append(dict_3d['Cz'])
            continue
        points_3d.append(dict_3d['E' + str(idx)])

    points_3d = np.transpose(np.asarray(points_3d))

    return points_2d, points_3d
