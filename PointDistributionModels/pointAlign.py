"""
This script aligns two shapes described by 3D point coordinates.

Our aim is to find a rigid transformation consisting of rotation and translation which transforms one shape
onto the other. The transformation should minimize sum of squared distances between shapes.
https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

Let P = {p1, p2, ...} and Q = {q1, q2, ...} be two sets of corresponding points in R^3. In optimization
problem we seek a rotation R and translation t such that argmin_R,t sum_i |(Rp_i + t) - q_i|^2

The optimization problem is divided into two sections:
Inner minimization with respect to translation points (tx, ty, tz)
Outer minimization with respect to theta (3D rotation)
"""

import numpy as np

from visualization_PDM import visualize_models


def align_point(ref_shape, shape, weight=None):
    """
    Finding the optimal transformation between given shapes (viz. documentation at the beginning).
    The optimal rotation matrix R np.array(3, 3) and the optimal translation vector t np.array(3, )

    :param weight: (optional)           Diagonal array of weights for each point pair np.array(3N, )
    :param ref_shape:                   Referential shape as vector of points np.array(3N, )
    :param shape:                       Shape to be aligned to reference shape as vector of points np.array(3N, )
                                           ... N is number of points

    :return: transformed                Transformed shape aligned with the referential one as np.array(3N, )
             R                          Optimal rotation matrix np.array(3, 3)
             t                          Optimal translation np.array(3)
    """
    # Initialize weights
    if weight is None:
        weight = np.ones(len(shape) // 3)

    # Compute centroids of both shape points
    ref_shape_centroid = compute_centroids(ref_shape, weight)
    shape_centroid = compute_centroids(shape, weight)

    # Compute centered vectors of both shapes
    ref_shape_centered = compute_centered_vector(ref_shape, ref_shape_centroid)
    shape_centered = compute_centered_vector(shape, shape_centroid)

    # Compute the optimal rotation and translation
    R = outer_min(ref_shape_centered, shape_centered)
    t = inner_min(R, ref_shape_centroid, shape_centroid)

    # Compute the aligned shape as np.array(3N, )
    shape_matrix = np.transpose(shape.reshape((int(len(shape) / 3), 3)))        # Reshape as np.array(3, N)
    transformed = (R @ shape_matrix) + t[:, None]
    transformed = np.transpose(transformed).reshape(-1)

    # visualize_models([ref_shape, shape])
    # visualize_models([ref_shape, np.transpose(transformed).reshape(-1)])

    return transformed, R, t


def compute_centroids(pts, weights=None):
    """
    This function computes centroid of given points. Centroids are average of points.

    :param pts:             3D points as vector np.array(3, N)
    :param weights:         Array of weights of points np.array(3N, )

    :return:                Centroid of given points as np.array(3, )
    """
    if weights is None:
        weights = np.ones(len(pts) // 3)

    pts = pts.reshape((int(len(pts) / 3), 3))
    centroid = np.array([np.sum(pts[:, 0] * weights),
                         np.sum(pts[:, 1] * weights),
                         np.sum(pts[:, 2] * weights)])
    centroid /= np.sum(weights)
    return centroid


def compute_centered_vector(pts, centroid):
    """
    This function computes centered vector of points from given points and centroid.

    :param pts:                 3D points as vector np.array(3N, )
    :param centroid:            Centroid of given points (can be computed by compute_centroids(pts)) np.array(3, )

    :return:                    Centered vector of 3D points np.array(3N, )
    """
    pts = pts.reshape((int(len(pts) / 3), 3))
    centered = pts - centroid
    return centered.reshape((-1))


def inner_min(R, ref_centroid, centroid):
    """
    Computation of the optimal translation minimizing the euclidean distance.

    :param R:                   Rotation matrix np.array(3, 3)
    :param ref_centroid:        Centroid of the referential shape np.array(3, )
    :param centroid:            Centroid of the shape to be aligned np.array(3, )

    :return:                    Optimal translation np.array(3, )
    """
    return ref_centroid - R @ centroid


def outer_min(ref_shape, shape):
    """
    Computation of the optimal rotation matrix R via singular value decomposition minimizing the euclidean distance.
    Warning: Both shapes have to be centered!

    :param ref_shape:           Vector of 3D referential centered points np.array(3N, )
    :param shape:               Vector of 3D centered points to be rotated np.array(3N, )

    :return:                    Optimal rotation matrix R which minimizes the least squares error between centered points
                                R represents the rotation from shape to referential shape ... ref_shape ~ R @ shape
    """
    N = int(len(ref_shape) / 3)
    shape = np.transpose(shape.reshape((N, 3)))
    ref_shape_t = ref_shape.reshape((N, 3))

    S = shape @ ref_shape_t
    U, _, Vt = np.linalg.svd(S)
    R = np.transpose(Vt) @ np.transpose(U)

    if np.linalg.det(R) < 0:
        V = np.transpose(Vt)
        V[:, -1] *= -1
        R = V @ np.transpose(U)

    return R
