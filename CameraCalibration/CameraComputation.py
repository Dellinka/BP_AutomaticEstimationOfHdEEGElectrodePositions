"""
This file computes camera projection matrices for each camera from solved dataset of computed 3d points.
Main purpose if this file is computation of average projection matrices used in next computations.
"""
import os

import numpy as np
from scipy.linalg import rq

from reader import load_points


def compute_camera_matrix(x, X):
    """
    This function computes camera projection matrix P from corresponding 3D world scene points
    and 2D image points using the Direct Linear Transformation (DLT) algorithm described in
    Multiple View Geometry in Computer Vision (page. 179)

    :param x        2D image points             ~ np.array(2, N) where N is number of points
    :param X        3D world scene points       ~ np.array(3, N) where N is number of points

    :returns P      camera projection matrix    ~ np.array(3,4)
    """
    def normalization(pts):
        """
        Normalize given coordinates - centroid (mean) to origin and root mean squared equals sqrt(2) (2D) / sqrt(3) (3D)
        viz. http://cmp.felk.cvut.cz/cmp/courses/Y33ROV/Y33ROV_ZS20092010/lab2/pointnorm.pdf
            :param pts              coordinates to normalize - np.array(D, number of points)

            :returns u2             normalized coordinates - np.array(D, number of points)
            :returns T              transformation matrix np array(D, D), u2 = Tu (where u and u2 are matrices in homogenous coordinates)
        """
        D = np.shape(pts)[0]
        pts = np.transpose(pts)
        if D != 2 and D != 3:
            print("Unexpected dimension " + str(D))
            exit(-1)

        centroid = np.mean(pts, 0)
        u2 = pts - centroid.T

        denominator = np.mean(np.sqrt(np.sum(u2 ** 2, axis=1)))
        scale = np.sqrt(2) / denominator if D == 2 else np.sqrt(3) / denominator
        normalized = scale * u2

        T = np.diag([scale, scale, 1]) if D == 2 else np.diag([scale, scale, scale, 1])
        T[:D, -1] = -scale * centroid

        return np.transpose(normalized), T

    _, points_num = np.shape(x)
    if points_num != np.shape(X)[1]:
        print("Number of corresponding points in 2D and 3D has to be equal!")
        return

    # Data normalization
    x, T = normalization(x)
    X, U = normalization(X)

    # From input points create homogeneous coordinates (point in a row and add column of ones)
    x = np.hstack((np.transpose(x), np.ones((points_num, 1))))
    X = np.hstack((np.transpose(X), np.ones((points_num, 1))))
    null = np.zeros(4)

    # Create set of equations in matrix M according to DLT algorithm
    M = np.array((2*points_num, 12))

    for i in range(0, points_num):
        xi, yi, wi = x[i][0], x[i][1], x[i][2]
        A = np.reshape(np.hstack((null, -wi * X[i], yi * X[i])), (1, 12))
        B = np.reshape(np.hstack((wi * X[i], null, -xi * X[i])), (1, 12))
        M = np.vstack((A, B)) if i == 0 else np.vstack((M, A, B))

    # Find minimal solution with SVD decomposition
    _, _, Vt = np.linalg.svd(M)
    V = np.transpose(Vt)
    P = (V[:, 11]).reshape((3, 4))

    return np.linalg.inv(T) @ P @ U


def KRC_from_P(P):
    """
    This function computes decomposition of the camera matrix P according to
    Multiple View Geometry in Computer Vision - page 163, chapter 6.2.4)

    :param P:       Camera projection matrix P  ~ np.array(3, 4)
                    P can be recomputed from the return parameter as P = K[R | -RC] (or P = K[R | t] where t = -RC)
                     - P = K @ (np.transpose(np.vstack((np.transpose(R), -R@C)))) * norm_coef

    :returns: K     The intrinsic matrix        ~ np.array(3, 3)
              R     The rotation matrix         ~ np.array(3, 3)
              C     The center of projection    ~ np.array(3, )
    """
    # Finding the camera center C (point where PC = 0)
    p1, p2, p3, p4 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    x = np.linalg.det(np.array([p2, p3, p4]))
    y = -np.linalg.det(np.array([p1, p3, p4]))
    z = np.linalg.det(np.array([p1, p2, p4]))
    t = -np.linalg.det(np.array([p1, p2, p3]))
    C = np.array([x/t, y/t, z/t])

    # K = [ αx  s   x0 ]
    #     [ 0   αy  y0 ]
    #     [ 0   0   1  ]
    # Where all diagonal entries are positive!
    M = P[0:3, 0:3]
    K, R = rq(M)
    norm_coef = float(K[2, 2])
    K = K / norm_coef

    if K[0, 0] < 0:
        K[:, 0] = -1 * K[:, 0]
        R[0, :] = -1 * R[0, :]

    if K[1, 1] < 0:
        K[:, 1] = -1 * K[:, 1]
        R[1, :] = -1 * R[1, :]

    return K, R, C


def compute_reprojected_points(X, P):
    """
    This function computes the reprojected 2D image points from
    3D scene points and the Camera projection matrix P.

    :param X        3D world scene points np.array(3, N)
    :param P:       Camera projection matrix P np.array(3, 4)

    :return:        Reprojected points as np.array(2, N)
    """
    # Create homogeneous coordinates from X and multiply with matrix P to get projected x in homogeneous coordinates
    _, points_num = np.shape(X)
    X = np.vstack((X, np.transpose(np.ones((points_num, 1)))))

    # Change projected points from homogeneous into normal coordinates
    x_proj = np.transpose(np.matmul(P, X))
    x_proj = np.transpose(np.transpose(x_proj) / x_proj[:, -1])
    x_proj = x_proj[:, 0:-1]

    return x_proj


def create_average_matrices(subjects, dir_2d, dir_3d, output_dir=os.path.join('..', 'Config', 'CameraMatrices')):
    """
    Compute camera matrices for each camera and subject and save average camera matrix for each camera.
    WARNING: Recomputation of camera matrices can change them a little due to rounding error.

    :param dir_2d                   Path to directory 2d_annotations_csv_orig
                                    (inside are all subjects and their 2d annotation in csv file (Sensor id | x | y))
    :param dir_3d                   Path to directory 3d_annotations_csv_orig
                                    (inside are all subjects and their 3d annotation in csv file (Sensor id | x | y | z))
    :param subjects                 Array with subjects names as string
    :param output_dir (optional)    Directory where to save camera matrices
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cam_num in range(1, 12):
        sum_matrix = np.zeros((3, 4))
        for s in subjects:
            file_2d = os.path.join(dir_2d, s, 'camera' + str(cam_num) + '.csv')
            file_3d = os.path.join(dir_3d, s + '.csv')
            pts_2d, pts_3d = load_points(file_2d, file_3d)

            P = compute_camera_matrix(pts_2d, pts_3d)
            if P[0][0] < 0:
                P = -P

            sum_matrix += P

        average_P = sum_matrix / len(subjects)
        np.savetxt(os.path.join(output_dir, 'camera' + str(cam_num) + '.out'), average_P)

    print("All camera matrices has been saved into {} directory".format(output_dir))


if __name__ == '__main__':
    input_dir_2d = os.path.join('..', 'Dataset', '2d_annotations_csv_orig')
    input_dir_3d = os.path.join('..', 'Dataset', '3d_annotations_csv_orig')

    subjects_ = ["AH", "DD", "EN", "HH", "JH", "JJ", "JK", "JR", "LK",
                "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "TN", "VV"]

    print("Purpose of this file is creation of average camera matrices from solved dataset. \n"
          "However the recomputation of camera projection matrices can change the computed matrices "
          "due to rounding error!\nIf you still want to continue, you have to uncomment the "
          "computation at the end of the CameraComputation.py file")
    # create_average_matrices(subjects_, input_dir_2d, input_dir_3d)
