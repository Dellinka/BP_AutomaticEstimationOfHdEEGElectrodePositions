import csv
import os
import numpy as np

from main import fit_and_save_model, create_and_save_3d_model
from statisticalModel import create_statistical_model

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


def save_all_models():
    """
    This function fits and saves all models for each subject.
    """
    idx = 1
    for subject in subjects:
        predicted_path = os.path.join('..', 'GeneratedData', 'Correspondences', subject, 'corresponding_points.csv')
        model_path = os.path.join('..', 'Config', 'StatisticalModel')
        output_dir = os.path.join('..', 'GeneratedData', 'FittedModels', subject)

        fit_and_save_model(predicted_file=predicted_path,
                           model_dir=model_path,
                           output_dir=output_dir)

        print("{} out of {} Done".format(idx, len(subjects)))
        idx += 1


def compute_all_coords():
    """
    This function computes and saves 3d coordinates for each subject.
    """
    idx = 1
    for subject in subjects:
        predicted_path = os.path.join('..', 'GeneratedData', 'Correspondences', subject, 'corresponding_points.csv')
        model_path = os.path.join('..', 'GeneratedData', 'FittedModels', subject)
        output_dir = os.path.join('..', 'GeneratedData', 'ComputedCoordinates', subject)

        create_and_save_3d_model(predicted_file=predicted_path,
                                 model_dir=model_path,
                                 output_dir=output_dir)

        print("{} out of {} Done".format(idx, len(subjects)))
        idx += 1


def evaluate_model(subject, computed_dir, original_dir=os.path.join('..', 'Dataset', '3d_annotations_csv_orig')):
    """
    Compute the average euclidean distance between original and predicted models

    :param computed_dir:        Directory to computed model file(~3d_model.out)
    :param original_dir:        Directory to original model file(~<subject>.csv)
    :param subject:             Name of subject, whose coordinates will evaluated
    :return: distance           Average euclidean distance between computed and original 3D coordinates
    """

    # Load original coordinates
    orig_coords = list()
    with open(os.path.join(original_dir, subject + '.csv')) as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            orig_coords.append(np.array([float(row[1]), float(row[2]), float(row[3])]))

    # Load predicted coodinates
    computed_coords = list()
    file = open(os.path.join(computed_dir, '3d_model.out'), 'r')
    for line in file.readlines():
        line = line.split()
        computed_coords.append(np.array([float(line[1]), float(line[2]), float(line[3])]))

    # Compute average euclidean distance between points
    dist = np.array(computed_coords) - np.array(orig_coords)
    norms = np.linalg.norm(dist, axis=1)
    return np.average(norms)


if __name__ == '__main__':
    # save_all_models()
    # compute_all_coords()
    subject = 'AH'
    dist = evaluate_model('AH', os.path.join('..', 'GeneratedData', 'ComputedCoordinates', subject))
    print("{}: The average euclidean distance between original and predicted model is {}".format(subject, dist))
