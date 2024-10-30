"""
This file contains various computation help functions used in PDM algorithm.
More information in functions documentation.
"""
import numpy as np
import copy

from visualization_PDM import visualize_models, visualize_correspondences


def PCA(C, alpha=0.95):
    """
    This script provides implementation of Principal component analysis algorithm.
    PCA is a process of computing the principal components and using them for changing the basis of the data. It is
    usually used for dimensionality reduction.

    We simplify the model by considering only the K largest eigen values. We choose the smallest K
    such that sum_K lambda_i >= alpha sum_N lambda_i
    Note: np.linalg.eigh(S) returns eigenvalues in ascending order

    :param C:               Covariance matrix np.array(3N, 3N)
    :param alpha:           Threshold for computing K number of eigen values

    :return: P              The most important eigenvectors of the model,
                            corresponding to the K largest eigenvalues np.array(3N, K)
             eig_values     K largest eigen values np.array(K, )
    """
    eig_values, P = np.linalg.eigh(C)

    limit = alpha * np.sum(eig_values)
    K = len(eig_values) - 1
    sum = eig_values[K]
    while K > 0:
        if sum > limit: break
        K -= 1
        sum += eig_values[K]

    P = np.fliplr(P[:, K + 1:])
    eig_values = np.flip(eig_values[K + 1:])
    return P, eig_values


def compute_assign_matrix_softassign(model_x, model_y, alpha, beta, x_colors=None, y_colors=None):
    """
    Compute weight matrix based on soft assign algorithm.

    :param model_x:                 3D referential model for weight matrix computation np.array(N, 3)
    :param model_y:                 3D model to be transformed for weight matrix computation np.array(M, 3)
    :param x_colors:                Colors of sensors of model x np.array(N, )
    :param y_colors:                Colors of sensors of model y np.array(M, )
    :param alpha:                   Threshold distance before points is treated as outlier float
    :param beta:                    Variable for deterministic annealing float

    :return: assign_matrix          Assign_matrix matrix with weight for each possible correspondence np.array(N, M)
    """
    # Compute distance matrix - distances between each two points
    distance_matrix = np.ones((len(model_x) + 1, len(model_y) + 1))
    for j in range(len(model_x)):
        for k in range(len(model_y)):
            distance_matrix[j, k] = np.linalg.norm(model_x[j] - model_y[k]) ** 2

    # Compute assign matrix as exp(-beta*Q)
    Q = distance_matrix - alpha
    assign_matrix = np.exp(- beta * Q)
    assign_matrix[:-1, -1] = 1
    assign_matrix[-1, :-1] = 1

    # Check colors of sensors, so black are assign on black only and white for white only
    if x_colors is not None and y_colors is not None:
        for row, color in enumerate(x_colors):
            assign_matrix[row, np.where(y_colors != color)] = 0

    return assign_matrix


def sinkhorns_method(matrix, epsilon=0.001, max_iterations=60):
    """
    According to Sinkhorn's theorem every square matrix with positive entries can be written as a
    Doubly stochastic matrix (sum_i a_ij = sum_j a_ij = 1). This should be a property of our assign matrix
    as well.
    Sinkhorn's method is iterative algorithm for finding such doubly stochastic matrix via iterative
    normalization through rows and columns of matrix.

    :param max_iterations:       Threshold for maximal number of iteration
    :param epsilon:             Threshold for convergence to be used for termination of sinkhorn loop
    :param matrix:              Assign matrix to be mapped to doubly stochastic np.array(N+1, M+1)
    :return:                    Doubly stochastic assign matrix
    """
    iter_num = 0
    diff_sum = epsilon + 1

    while diff_sum > epsilon and iter_num < max_iterations:
        prev_matrix = copy.deepcopy(matrix)

        # Column normalization
        cols_sum = np.sum(matrix, 0)
        cols_sum[-1] = 1
        matrix /= cols_sum

        # Rows normalization
        rows_sum = np.sum(matrix, 1)
        rows_sum[-1] = 1
        rows_sum = rows_sum.reshape((len(rows_sum), 1))
        matrix /= rows_sum

        diff_sum = np.sum(np.abs(prev_matrix - matrix))
        iter_num += 1

    return matrix


def compute_correspondences_from_soft_assignment(assign_matrix):
    """
    This function computes corresponding points from assign matrix.

    :param assign_matrix:       Weight/Assign matrix from soft assign algorithm np.array(N, M)
                                where N is number of predicted points and M is number of model

    :return:                    Returns indexes(*3 - indexes of pred, model from main) of
                                predicted and corresponding model points
    """
    idx_pred = list()
    idx_model = list()
    pred_pts_num, model_pts_num = assign_matrix.shape

    for pred_i in range(pred_pts_num - 1):
        model_i = np.argmax(assign_matrix[pred_i])

        if model_i == model_pts_num - 1:
            continue  # Slack variable

        if np.argmax(assign_matrix[:, model_i]) == pred_i:
            idx_pred.append(3 * pred_i)
            idx_pred.append(3 * pred_i + 1)
            idx_pred.append(3 * pred_i + 2)
            idx_model.append(3 * model_i)
            idx_model.append(3 * model_i + 1)
            idx_model.append(3 * model_i + 2)

    return np.array(idx_pred), np.array(idx_model)


def compute_distance_matrix(model_x, model_y):
    """
    This function computes distance matrix between two models.

    :param model:               3D Model as np.array(M, 3)
    :param predicted:           3D Model as np.array(N, 3)

    :return:                    Distance matrix size as np.array(M, N)
    """
    distance_matrix = np.ones((len(model_x), len(model_y)))
    for j in range(len(model_x)):
        for k in range(len(model_y)):
            distance_matrix[j, k] = np.linalg.norm(model_x[j] - model_y[k])

    return distance_matrix


def distance_between_correspondences(predicted, model, assign_matrix, title="", save_f="", visualize=False):
    """
    This function computes distance between predicted and transformed correspondence data points
    according to given assignment matrix.

    :param predicted:               3D model of predicted points as np.array(M, 3)
    :param model:                   3D model of transformed points as np.array(N, 3)
    :param assign_matrix:           Assignment matrix between predictions and model np.array(M+1, N+1)
    :param title:                   Optional title for visualization
    :param visualize:               Optional boolean if visualize points or not
    """
    idx_predicted, idx_model = compute_correspondences_from_soft_assignment(assign_matrix)
    if len(idx_predicted) == 0: return -1
    pred_corresponding = predicted.reshape(-1)[idx_predicted]
    model_corresponding = model.reshape(-1)[idx_model]
    pred_corresponding = pred_corresponding.reshape((int(len(pred_corresponding) / 3), 3))
    model_corresponding = model_corresponding.reshape((int(len(model_corresponding) / 3), 3))
    dist = round(np.average(np.linalg.norm(pred_corresponding - model_corresponding, axis=1)), 3)

    # Visualize
    if visualize:
        matched = len(model_corresponding)
        if len(save_f) != 0:
            save_corresp = save_f + "_corresp.png"
            save_models = save_f + "_models.png"
        else:
            save_corresp = ""
            save_models = ""
        visualize_correspondences(pred_corresponding.reshape(-1), model_corresponding.reshape(-1),
                                  labels=["Predicted corresponding", "Model corresponding"],
                                  title=title + " Matched {} Distance {}".format(matched, dist),
                                  save_f=save_corresp)
        visualize_models([predicted.reshape(-1), model.reshape(-1)], labels=["Predicted", "Model"],
                         title=title, save_f=save_models)

    return dist


def labels2colors(labels):
    """
    According to given labels and BLACK_SENSORS return array with corresponding sensor color.
    0 for black and 1 for white

    :param labels:          Labels/names of sensors as 'E<number>' or 'Fid<type>' np.array(M, )

    :return:                Colors of corresponding sensors np.array(M, )
    """
    # According to https://ars.els-cdn.com/content/image/1-s2.0-S0022249614000571-gr11.jpg + 'FidNZ' (fiducial == E31)
    BLACK_SENSORS = ['FidNz', 'E31', 'E36', 'E67', 'E72', 'E111', 'E114', 'E119', 'E168', 'E173', 'E199',
                     'E219', 'E224', 'E234', 'E237', 'E244', 'E247', 'Cz']

    colors = np.ones(len(labels))

    for sensor in BLACK_SENSORS:
        colors[np.where(labels == sensor)] = 0

    return colors
