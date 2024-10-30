"""
Purpose of this file is computation of optimal pose (R, t) and shapes (b) parameters.
Main idea implemented of this algorithm (in optimal_Rtb() function) is based on algorithm for
fitting a 2D model to new points described in Statistical Models of Appearance for Computer Vision
(chapter 4.8, page 22) from T.F.Cootes and C.J.Taylor from 2004
Online version of this book https://www.face-rec.org/algorithms/AAM/app_models.pdf

Computation of the optimal rotation and translation parameters is based on two research papers
first one Estimating 3-D Location Parameters Using Dual Number Quaternions (M.W.Walker and L.Shao)
Online version https://deepblue.lib.umich.edu/bitstream/handle/2027.42/29059/0000092.pdf?sequence=1
and second NEW ALGORITHMS FOR 2D AND 3D POINT MATCHING:POSE ESTIMATION AND CORRESPONDENCE (S. Gold et. al)
Online version https://cmp.felk.cvut.cz/~amavemig/softassign.pdf - I thing this one is wrong
"""
import copy
import numpy as np

from computationFunctions_PDM import compute_correspondences_from_soft_assignment


def compute_Rt_walker(model, referential, w):
    """
    Compute optimal rotation and translation from x to y using Walker et al.’s method.
    Minimize function sum_ij w_ij |x_i - t - Ry_j|^2

    R = W(r)^t * P(r) and t = W(r)^t * s

    Computation according to https://deepblue.lib.umich.edu/bitstream/handle/2027.42/29059/0000092.pdf?sequence=1
    The other research paper is wrong!

    :param model:           3D model of model points to be transformed np.array(N, 3)
    :param referential:     3D model of measured points as referential model np.array(M, 3)
    :param w:               Weight/assign matrix computed using soft assign (N+1, M+1)
    :return: R, t           Optimal parameters of rotation np.array(3, 3) and translation np.array(3, )
    """
    def compute_P(pt):
        r1, r2, r3, r4 = pt[0], pt[1], pt[2], pt[3] if len(pt) > 3 else 0
        return np.array([[ r4, -r3,  r2, r1],
                         [ r3,  r4, -r1, r2],
                         [-r2,  r1,  r4, r3],
                         [-r1, -r2, -r3, r4]])

    def compute_W(pt):
        r1, r2, r3, r4 = pt[0], pt[1], pt[2], pt[3] if len(pt) > 3 else 0
        return np.array([[ r4,  r3, -r2, r1],
                         [-r3,  r4,  r1, r2],
                         [ r2, -r1,  r4, r3],
                         [-r1, -r2, -r3, r4]])

    c1, c2, c3 = np.zeros((4, 4)), 0, np.zeros((4, 4))
    for j in range(len(model)):
        for k in range(len(referential)):
            P = compute_P(referential[k])
            W = compute_W(model[j])

            c1 += w[k, j] * P.transpose() @ W
            c2 += w[k, j]
            c3 += w[k, j] * (W - P)
    c1 = -2 * c1
    c2 = c2 * np.eye(4)
    c3 = 2 * c3

    eig_values, eig_vectors = np.linalg.eig(1/2 * (c3.transpose() @ np.linalg.inv(c2 + c2.transpose()) @ c3 - c1 - c1.transpose()))
    r = eig_vectors[:, np.argmax(eig_values)]
    s = - 2 * np.linalg.inv(c2 + c2.transpose()) @ c3 @ r

    Wt_r = np.transpose(compute_W(r))
    R = Wt_r @ compute_P(r)
    t = Wt_r @ s
    return R[:-1, :-1], t[:-1]


def optimal_Rtb(mean, predicted, assign_matrix, phi, phi_eigval, epsilon=0.0001):
    """
    Computation of optimal R, t and b parameters based on algorithm for fitting a model to new points
    from Statistical Models of Appearance for Computer Vision (chapter 4.8, page 23).

    :param mean:                        Mean 3D model to be transformed np.array(M, 3)
    :param predicted:                   Predicted 3D model on which mean model should be transformed np.array(N, 3)
    :param assign_matrix:               Weight/Assign matrix from softassign np.array(N, M)
    :param phi:                         Eigen vectors from PCA algorithm np.array(3M, K)
    :param phi_eigval:                  Eigen values corresponding to phi np.array(K, )
    :param epsilon:                     Optional threshold for checking convergence of b parameter

    :return:
    """
    b = np.zeros(phi.shape[1])
    idx_predicted, idx_model = compute_correspondences_from_soft_assignment(assign_matrix)

    while True:
        # Store shape parameters for convergence check
        prev_b = copy.deepcopy(b)

        # Create model
        model = mean.reshape(-1) + phi @ b
        model = model.reshape((int(len(model) / 3), 3))

        # Find optimal pose parameters
        R, t = compute_Rt_walker(model, predicted, assign_matrix)

        # Project prediction into model coordinates
        y = (np.transpose(np.linalg.inv(R) @ np.transpose(predicted) - t[:, None])).reshape(-1)

        # According to assign matrix find correspondences and update b parameter
        x = mean.reshape(-1)[idx_model]
        y = y.reshape(-1)[idx_predicted]
        phi_ = phi[idx_model]

        # Apply constraints on b such as all b_i are in range +- 3*sqrt(lambda_i)
        b = np.transpose(phi_) @ (y - x)
        for idx in range(len(b)):
            tmp = 3 * np.sqrt(phi_eigval[idx])
            if tmp < b[idx]:
                b[idx] = tmp
            elif -tmp > b[idx]:
                b[idx] = -tmp

        # Check convergence of b parameter
        diff_sum = np.sum(np.abs(prev_b - b))
        if diff_sum < epsilon:
            return R, t, b
