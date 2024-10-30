"""
This file creates a point distribution model from training 3D model examples. According to PDM viz.
http://cmp.felk.cvut.cz/cmp/courses/33DZOzima2007/slidy/pointdistributionmodels.pdf (slide 13).

This algorithm aims to create statistical model for our data described by mean
of training shapes and small number of eigenvectors. We have one referential
shape and all others can be obtained as x = x_mean + P*b, where x_mean is defined
as the mean across all training shapes, P is matrix of the most important eigen
vectors of the model, corresponding to the K largest eigenvalues and b is a vector
of scaling values for each principal component (it should be within +- 3 standard
deviations).
"""
import copy
import os
import numpy as np

from IOhelper import read_points, save_statistical_model
from computationFunctions_PDM import PCA
from pointAlign import align_point, compute_centroids
from visualization_PDM import visualize_models

DATASET_3D = os.path.join('..', 'Dataset', '3d_annotations_csv_orig')
visualize_aligning = False
visualize_approxim = False
visualize_idx = 14


def create_statistical_model(subjects, dataset_3D=DATASET_3D):
    """
    Computation of statistical model from solved 3d coordinates from dataset.
    (More information at the beginning of this file.)

    :param subjects                 Array with name of subjects as string
    """
    # Read all 16 models from dataset and save them into all_shapes matrix (points from one dataset into one column)
    all_shapes = list()
    points, labels = read_points(os.path.join(dataset_3D, subjects[0] + '.csv'))
    all_shapes.append(points)

    for s in subjects:
        if s == subjects[0]: continue
        points, _ = read_points(os.path.join(dataset_3D, s + '.csv'))
        all_shapes.append(points)

    all_shapes = np.transpose(np.asarray(all_shapes))
    if visualize_aligning:
        visualize_models([all_shapes[:, 0], all_shapes[:, visualize_idx-1]],
                         labels=["Referential model", "Model before aligning"],
                         title="Referential and second model before aligning")
                         # save_f=os.path.join('..', 'Images', 'PointDistributionModels', 'before_aligning.png'))

    # Align all shapes with the first one and calculate the mean by averaging the transformed shapes.
    N, M = all_shapes.shape         # N .. number of points*3, M .. number of training shapes
    for i in range(1, M):
        all_shapes[:, i], _, _ = align_point(all_shapes[:, 0], all_shapes[:, i])
    x_mean = np.mean(all_shapes, 1)

    if visualize_aligning:
        visualize_models([all_shapes[:, 0], all_shapes[:, visualize_idx-1]],
                         labels=["Referential model", "Model after aligning"],
                         title="Referential and second model after first aligning")
                         # save_f=os.path.join('..', 'Images', 'PointDistributionModels', 'before_aligning.png'))

    # Iterate until the difference between old and new mean shape is smaller than predefined threshold.
    diff = np.inf
    while diff > 1e-6:
        # Align the mean shape to referenced shape (first shape from all_shapes, necessary for convergence),
        # align all other shapes to adjusted mean and recalculate mean. Repeat until convergence.
        x_mean, _, _ = align_point(all_shapes[:, 0], x_mean)
        for i in range(1, M):
            all_shapes[:, i], _, _ = align_point(x_mean, all_shapes[:, i])

        old_mean = x_mean
        x_mean = np.mean(all_shapes, 1)
        diff = np.linalg.norm((old_mean-x_mean) / N)

    # Calculation of the covariance matrix
    S = np.zeros((N, N))
    for i in range(M):
        delta = all_shapes[:, i] - x_mean
        S += delta[:, None] * delta
    S = S / M

    if visualize_approxim:
        # Visualization
        P, eig_val = PCA(S)
        x = all_shapes[:, visualize_idx-1]
        b = np.transpose(P) @ (x - x_mean)
        x_app = x_mean + P @ b

        visualize_models([x, x_mean],
                         labels=["Original model", "Mean model"],
                         title="Comparison of original and mean model")

        visualize_models([x, x_app],
                         labels=["Original model", "Approximated model"],
                         title="Comparison of original and approximated model, distance")

    output_dir = os.path.join('..', 'Config', 'StatisticalModel')
    save_statistical_model(x_mean.reshape((int(len(x_mean) / 3), 3)), labels, output_dir=output_dir, cov_matrix=S)
    print("Statistical model saved into {} directory".format(output_dir))


def compute_approximated_model(subject, dataset_3D=DATASET_3D, visualize=False):
    """
    Computes the approximated model for given subject using the statistical model from configuration.

    :param subject:             Name of the subject which model will be computed
    :param visualize:           Boolean value for visualization

    :return: pts                Original model
             pts_app            Approximated model
    """
    # Read models
    model_dir = os.path.join('..', 'Config', 'StatisticalModel')
    pts, _ = read_points(os.path.join(dataset_3D, subject + '.csv'))
    x_mean = np.loadtxt(os.path.join(model_dir, 'model.out'))
    S = np.loadtxt(os.path.join(model_dir, 'covariance_matrix.out'))

    # Shift mean model to predicted one
    shift = compute_centroids(x_mean.reshape(-1)) - compute_centroids(pts.reshape(-1))
    x_mean = (x_mean - shift).reshape(-1)

    P, eig_val = PCA(S)
    b = np.zeros(P.shape[1])
    diff_sum = 1

    while diff_sum > 0.00001:
        # Store shape parameters for convergence check
        prev_b = copy.deepcopy(b)

        # Create model
        model = x_mean.reshape(-1) + P @ b
        model = model.reshape((int(len(model) / 3), 3))

        # Find optimal pose parameters
        _, R, t = align_point(pts, model.reshape(-1))

        # Project prediction into model coordinates
        y = (np.transpose(np.linalg.inv(R) @ np.transpose(pts.reshape(int(len(pts)/3), 3)) - t[:, None])).reshape(-1)

        # Apply constraints on b such as all b_i are in range +- 3*sqrt(lambda_i)
        b = np.transpose(P) @ (y - x_mean)
        for idx in range(len(b)):
            tmp = 3 * np.sqrt(eig_val[idx])
            if tmp < b[idx]:
                b[idx] = tmp
            elif -tmp > b[idx]:
                b[idx] = -tmp

        # Check convergence of b parameter
        diff_sum = np.sum(np.abs(prev_b - b))

    pts_app, _, _ = align_point(pts, (x_mean.reshape(-1) + P @ b).reshape(-1))

    if visualize:
        visualize_models([pts, pts_app],
                         labels=["Original model", "Approximated model"],
                         title="Comparison of original and approximated model")

    return pts, pts_app


def fitting_new_points(subjects_all):
    """
    Algorithm for model fitting on new data, the of average euclidean distance
    between original and fitted model is computed

    :param subjects_all         The names of subjects from training dataset
    """
    err = list()
    for _ in range(len(subjects_all)):
        subject = subjects_all[0]
        subjects_all.pop(0)
        # print("Statistical model on subjects ", str(subjects_all))
        create_statistical_model(subjects_all)
        # print("Approximated model of subject ", subject)
        orig, approxim = compute_approximated_model(subject, visualize=True)
        dist = np.average(np.linalg.norm(orig.reshape((int(len(orig)/3), 3))-approxim.reshape((int(len(approxim)/3), 3)), axis=1))
        print("Euclidean distance between original and approximated models subject {}: {}".format(subject, dist))
        subjects_all.append(subject)
        err.append(dist)
        print("----------------------------")

    print("Average distance is {}".format(np.average(np.asarray(err))))


if __name__ == '__main__':
    subjects_ = ["AH", "DD", "EN", "HH", "JH", "JJ", "JK", "JR", "LK",
                "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "TN", "VV"]
    print("Purpose of this file is creation of statistical 3d model from solved dataset. \n"
          "The recomputed statistical model can change a little due to rounding error!\n"
          "If you still want to continue, you have to uncomment the "
          "computation at the end of the statisticalModel.py file.")
    # create_statistical_model(subjects_)