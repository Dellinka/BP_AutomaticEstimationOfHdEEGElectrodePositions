"""
We have partial predicted 3D model from previous codes. Our aim is to fit statistical
model (computed from training data and find final 3D model in statisticalModel.py) onto predicted partial 3D model.

Description of fit_and_save_model() algorithm
    1)  Shift statistical model to predicted one so they have the same centroid (shifted model is called mean model next)
    2)  Compute the assign matrix of this two models (computation of assign matrix inspired by this
        research paper https://cmp.felk.cvut.cz/~amavemig/softassign.pdf but added color constraints)
    3)  Iterate until convergence of distance between predicted and mean model
            - compute optimal R, t, b parameter for given assign matrix and recompute mean model
            3.b) Iterate until convergence of distance between assign matrices (fixed b parameter)
                - compute new assign matrix
                - compute optimal R, t parameters using Walker et. al algorithm (viz R and t computation function)
                - Recompute mean model

"""
import copy
import os
import numpy as np

from IOhelper import read_predicted_model, save_statistical_model, save_computed_coordinates
from computationFunctions_PDM import compute_assign_matrix_softassign, PCA, \
    sinkhorns_method, distance_between_correspondences, \
    compute_distance_matrix, labels2colors
from optimalRtb import optimal_Rtb, compute_Rt_walker
from pointAlign import compute_centroids
from visualization_PDM import visualize_models


def fit_and_save_model(predicted_file, model_dir, output_dir):
    """
    This function fits statistical model onto predicted one as described at the beginning of file.
    Fitted model saves into output file from kwargs arguments.

    :param predicted_file            Path to file with predicted model (~/corresponding_points.csv)
    :param model_dir                 Path to directory in Config with information about statistical model
    :param output_dir                Path to output directory
    """
    global R, t, b, assign_matrix
    # Read data from files
    pred_model, pred_colors = read_predicted_model(predicted_file)
    x_mean = np.loadtxt(os.path.join(model_dir, 'model.out'))
    x_mean_labels = np.genfromtxt(os.path.join(model_dir, 'model_labels.out'), dtype='str')
    x_mean_colors = labels2colors(x_mean_labels)
    cov_matrix = np.loadtxt(os.path.join(model_dir, 'covariance_matrix.out'))

    # Shift mean model to predicted one
    shift = compute_centroids(x_mean.reshape(-1)) - compute_centroids(pred_model.reshape(-1))
    x_mean = x_mean - shift

    # Define all variable needed for soft assign algorithm
    alpha = np.max((np.std(compute_distance_matrix(x_mean, pred_model)), 1))  # threshold distance before points is treated as outlier
    beta = 0.6
    phi, phi_eigval = PCA(cov_matrix)
    epsilon_assign = 0.01
    epsilon_dist = 0.001

    # Define first assign matrix for x_mean and predicted model
    assign_matrix = compute_assign_matrix_softassign(pred_model, x_mean, alpha, beta, pred_colors, x_mean_colors)
    assign_matrix = sinkhorns_method(assign_matrix)
    if distance_between_correspondences(pred_model, x_mean, assign_matrix, visualize=False) == -1:
        print("No correspondences for beta ", beta)
        exit(-1)

    # Prepare optimal parameters
    model_opt = x_mean
    assign_matrix_opt = assign_matrix
    dist_opt = 9999

    iter_b = 0
    diff_dist = 1
    dist = 0

    # Iter until convergence of distance between predicted partial model and statistical model
    while diff_dist > epsilon_dist and iter_b < 8:
        R, t, b = optimal_Rtb(x_mean, pred_model, assign_matrix, phi, phi_eigval)
        model = x_mean.reshape(-1) + phi @ b
        model = model.reshape((int(len(model) / 3), 3))
        model = np.transpose(R @ np.transpose(model) + t[:, None])
        diff_assign, iter_n = 1, 0
        iter_b += 1

        # Soft assign loop - for fixed shape (b) finds optimal R and t and assign matrix
        while diff_assign > epsilon_assign and iter_n < 50:
            iter_n += 1
            prev_assign_matrix = copy.deepcopy(assign_matrix)
            assign_matrix = compute_assign_matrix_softassign(pred_model, model, alpha, beta, pred_colors,
                                                             x_mean_colors)
            assign_matrix = sinkhorns_method(assign_matrix)

            # Compute optimal R, t using computed assign matrix
            R, t = compute_Rt_walker(model, pred_model, assign_matrix)
            model = np.transpose(R @ np.transpose(model) + t[:, None])

            diff_assign = np.linalg.norm(prev_assign_matrix - assign_matrix)

        # Compute average distance between correspondences
        prev_dist = copy.deepcopy(dist)
        dist = distance_between_correspondences(pred_model, model, assign_matrix, title="Iter number {}.".
                                                format(iter_b), visualize=False)
        print("Optimal assign matrix for {}. b parameter found after {} iterations with distance {}".
              format(iter_b, iter_n, dist))

        # Check if it is the best model
        if dist_opt > dist:
            dist_opt = copy.deepcopy(dist)
            model_opt = copy.deepcopy(model)
            assign_matrix_opt = copy.deepcopy(assign_matrix)

        diff_dist = abs(prev_dist - dist)

    dist = distance_between_correspondences(pred_model, model_opt, assign_matrix_opt)
    visualize_models([pred_model.reshape(-1), model_opt.reshape(-1)], labels=["Predicted model", "Fitted model"],
                     title="Comparison of predicted coordinates and fitted statistical model \n (Distance={})".format(dist),
                     save_f=os.path.join(output_dir, "pred_vs_fitted.png"))
    model_labels = np.genfromtxt(os.path.join(model_dir, 'model_labels.out'), dtype='str'),
    save_statistical_model(model_opt, model_labels, output_dir=output_dir,
                           assign_matrix=assign_matrix_opt)


def create_and_save_3d_model(predicted_file, model_dir, output_dir):
    """
    Computes all 3d coordinates from fitted model which were not predicted before.

    :param predicted_file            Path to file with fitted model (~/corresponding_points.csv)
    :param model_dir                 Path to directory with information about fitted model
    :param output_dir                Path to output directory
    """
    # Read all data (optimal fitted model and predicted model)
    predicted, _ = read_predicted_model(predicted_file)
    model = np.loadtxt(os.path.join(model_dir, 'model.out'))
    model_labels = np.genfromtxt(os.path.join(model_dir, 'model_labels.out'), dtype='str')
    assignments = np.loadtxt(os.path.join(model_dir, 'assign_matrix.out'))
    final_coords = {}

    # Find corresponding points according to assigment and save them
    pred_pts_num, model_pts_num = assignments.shape
    for pred_i in range(pred_pts_num - 1):
        model_i = np.argmax(assignments[pred_i])

        if model_i == model_pts_num - 1:
            continue  # Slack variable

        if np.argmax(assignments[:, model_i]) == pred_i:
            final_coords[model_labels[model_i]] = predicted[pred_i]
            # print("{} = {} (predicted idx {})".format(model_labels[model_i], predicted[pred_i], pred_i))

    # Other points compute from model
    for idx, label in enumerate(model_labels):
        if label in final_coords: continue
        final_coords[label] = model[idx]
        # print("{} = Computed as {})".format(label, model[idx]))

    save_computed_coordinates(model_labels, final_coords, output_dir=output_dir)
