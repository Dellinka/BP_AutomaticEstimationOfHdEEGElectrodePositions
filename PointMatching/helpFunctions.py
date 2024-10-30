"""
In this file there are some help functions mainly for reading and saving data.
"""
import csv
import os
import numpy as np
import CorrespondingPoints


def read_predictions(file):
    """
    Read predictions from csv file (in format of output from template matching) and
    returns predicted points and their colors

    :param file:            Path to file with predictions (from template matching)
    :return:                Predicted points as dictionary
                            {(x, y): color}
    """
    with open(file) as file_content:
        csv_reader = csv.reader(file_content, delimiter=';')
        points = {}
        index = 0

        for row in csv_reader:
            if index == 0:
                index += 1
                continue

            bbox = row[1].split()
            ymin, xmin, ymax, xmax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x_center = xmin + (xmax - xmin) / 2
            y_center = ymin + (ymax - ymin) / 2

            points[(x_center, y_center)] = int(row[2])

    return points


def read_camera_constraints(file):
    """
    Read constrains from camera and save it into dictionary of sets.

    :param file:                Path to file with camera constraints

    :return: constraints        Dictionary of sets with camera constraints.
                                {1: {2, 3, 4, 5, 6}, 2: {...}, ...}
    """
    constraints = {}

    f = open(file, "r")
    lines = f.readlines()

    for l in lines:
        constraint = set()
        idx, cameras = l.strip().split(':')
        cameras = cameras.split()

        for c in cameras:
            constraint.add(int(c))
        constraints[int(idx)] = constraint

    f.close()
    return constraints


def save_correspondences(correspondences, output_file):
    """
    Save final correspondences into csv file.

    :param correspondences:         Array of CorrespondingPoints to be saved
    :param output_file:             Path to output file
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    f = open(output_file, 'w')
    f.write('3D coordinate; Color; Camera; 2D coordinate;')
    for corresponding in correspondences:
        line = '\n' + str(corresponding.coord_3d[0]) + ' ' \
                    + str(corresponding.coord_3d[1]) + ' ' \
                    + str(corresponding.coord_3d[2]) + '; ' \
                    + str(corresponding.color) + '; '

        for camera, point in corresponding.points.items():
            line += str(camera) + '; ' + \
                    str(point[0]) + ' ' + str(point[1]) + '; '

        f.write(line)
    f.close()


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

            points = CorrespondingPoints.CorrespondingPoints(int(row[1]))
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
