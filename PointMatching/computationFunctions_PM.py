import numpy as np
import gurobipy as gp
from gurobipy import GRB


def compute_fundamental_matrix(P1, P2):
    """
    This function computes fundamental matrix according to Multiple view geometry book
    https://github.com/jmmanley/VGG-Multiple-View-Geometry/blob/master/vgg_multiview/vgg_F_from_P.m

    :param P1:          Projection camera matrix from first camera
    :param P2:          Projection camera matrix from second camera
    :return:            Fundamental matrix F for given cameras
                        Such as l = Fx1 (where x1 is point in image taken with first camera)
                        and l is epipolar line in second image on which lies point x2
    """
    x1 = np.array((P1[1], P1[2]))
    x2 = np.array((P1[2], P1[0]))
    x3 = np.array((P1[0], P1[1]))
    y1 = np.array((P2[1], P2[2]))
    y2 = np.array((P2[2], P2[0]))
    y3 = np.array((P2[0], P2[1]))

    F = np.array([
        [np.linalg.det(np.vstack((x1, y1))), np.linalg.det(np.vstack((x2, y1))), np.linalg.det(np.vstack((x3, y1)))],
        [np.linalg.det(np.vstack((x1, y2))), np.linalg.det(np.vstack((x2, y2))), np.linalg.det(np.vstack((x3, y2)))],
        [np.linalg.det(np.vstack((x1, y3))), np.linalg.det(np.vstack((x2, y3))), np.linalg.det(np.vstack((x3, y3)))]])

    return F


def find_possible_correspondences(pointsA, pointsB, matrixA, matrixB, epsilon):
    """
    For each predicted point from pointsA find set of pointsB which are consistent
    with this point => has the same color (label) and reprojection error is smaller than epsilon.
    The reprojection error refers to distance from epipolar line.

    :param epsilon:         Threshold for reprojection error - determines consistent points (in px)
    :param pointsA:         Predicted points from camera A as dictionary {(x, y): color , ...}
    :param pointsB:         Predicted points from camera B as dictionary {(x, y): color , ...}
    :param matrixA:         Projective camera matrix from camera A np.array (3, 4)
    :param matrixB:         Projective camera matrix from camera B np.array (3, 4)

    :return: consistent                Array with all possibly corresponding points in two provided predictions and cameras
                                       Corresponding points are saved as as dictionary where point from A (as tuple) is key and
                                       its value is array of corresponding points from B (as tuple)
                                       {((xA, yA), color): [(xB, yB), (xB, yB), ...], ...}
    """
    consistent = {}
    F = compute_fundamental_matrix(matrixA, matrixB)

    for centerA, colorA in pointsA.items():
        key = (centerA, colorA)
        consistent[key] = list()
        line = F @ np.hstack((np.asarray(centerA), 1))  # Computation of epipolar line l = Fx (x in homogeneous coords)
        a, b, c = line[0], line[1], line[2]     # Parameters for epipolar line ax + by + cz = 0 (in image B)
        denominator = np.sqrt(a*a + b*b)    # For point distance computation -> | a*x + b*y + c | / sqrt( a*a + b*b)

        for centerB, colorB in pointsB.items():
            if colorA != colorB: continue
            distance = np.abs(a*centerB[0] + b*centerB[1] + c) / denominator
            if distance <= epsilon:
                consistent[key].append(centerB)

    return consistent


def linear_triangulation(x1, x2, P1, P2):
    """
    Computes 3D coordinate from two corresponding points and related camera matrices according to
    Linear triangulation method described in Multiple View Geometry in Computer Vision (page 312, chapter 12.2).
    Computation of matrix A =  [ x(p3t) - (p1t) ]               ... where pit and p'it are rows of
                               [ y(p3t) - (p2t) ]                   camera matrices P and P1
                               [ x'(p'3t) - (p'1t) ]            ... Then find 3d point X from AX = 0
                               [ y'(p'3t) - (p'2t) ]

    :param x1:              2d point coordinate in first picture np.array(2,)
    :param x2:              2d point coordinate in second picture np.array(2,)
    :param P1:              Camera matrix of the first camera np.array(3,4)
    :param P2:              Camera matrix of the second camera np.array(3,4)

    :return:                3d coordinate computed from given parameters np.array(3,)
    """
    A = np.vstack((x1[0] * P1[2] - P1[0],
                   x1[1] * P1[2] - P1[1],
                   x2[0] * P2[2] - P2[0],
                   x2[1] * P2[2] - P2[1]))

    _, _, Vt = np.linalg.svd(A)
    V = np.transpose(Vt)
    X = V[:, 3] / V[-1, -1]

    return X[0:-1]


def compute_reprojected_point(X, P):
    """
    Computes 2D image coordinate from 3D scene coordinate and projection matrix P.

    :param X:               Point in 3d coordinates np.array(3,)
    :param P:               Camera matrix np.array(3,4)

    :return:                Point in 2d coordinates as tuple (x, y)
    """
    X = np.hstack((X, 1))
    x = np.matmul(P, X.reshape((4, 1)))
    x = x / x[-1]
    x = tuple(np.transpose(x)[0][0:-1])
    return tuple((round(x[0], 2), round(x[1], 2)))


def find_consistent_from_reprojected(point, color, predictions, epsilon):
    """
    Decides if reprojected point is consistent - has reprojection error smaller than epsilon and
    has the right color. If yes, return predicted consistent point else None

    :param point:                       Reprojected point np.array(2,)
    :param color:                       Right color of sensor (0 - black, 1 - white)
    :param predictions:                 All predicted points in image dictionary{(x, y): color, ...}
    :param epsilon:                     Threshold for reprojection error
    :return:
    """
    if color == 0:
        # Smaller probability of another black sensor in the epsilon area
        epsilon = 2*epsilon

    centers = list(predictions)
    distances = np.array([np.linalg.norm(np.array(key)-point) for key in centers])
    closest_idx = int(np.argmin(distances))

    if distances[closest_idx] > epsilon:
        return None

    if predictions[centers[closest_idx]] != color:
        return None

    return centers[closest_idx]


def find_common_subsets(cameras, constraints):
    """
    Find all subsets of camera views through cameras according to given camera constraints.

    :param cameras:             Set of number of cameras for subsets computation {1, 2, 3, ...}
    :param constraints:         Camera constraints (which camera see which view)

    :return:                    Array of all subsets, where all cameras in subset see same view (according to constraints)
    """
    def recursion(arr, const, all_const):
        if len(const) == 0: return []
        res = []
        for idx, value in enumerate(arr):
            if value not in const:
                continue

            res.append([value])
            if idx+1 >= len(arr): return res
            ret = recursion(arr[idx+1:], const.intersection(all_const[value]), all_const)
            res = res + [[value] + c for c in ret]

        return res

    return recursion(list(cameras), cameras, constraints)


def find_best_subset_greedy(all_correspondences):
    """
    Find best subset from all correspondences (best sets 'C' from 'M') via greedy algorithm.

    :param all_correspondences:             All consistent points list of CorrespondingPoints

    :return:                                Best subset as list of dictionaries with points
                                            [{camera: (x, y), camera2: (x, y), ...}, {...}, ...]
    """
    final = list()
    all_correspondences = sorted(all_correspondences, key=lambda x: len(x.points), reverse=True)

    while all_correspondences:
        to_remove = all_correspondences.pop(0)
        final.append(to_remove)

        for camera, point in to_remove.points.items():
            # print("Searching for camera: {}, point: {}". format(camera, point))
            for idx, corresponding in enumerate(all_correspondences[:]):
                if point == corresponding.points.get(camera):
                    # print("--- Deleted {}".format(all_correspondences[idx].points))
                    all_correspondences.remove(corresponding)

    return final


def find_best_subset_indexes_gurobi(all_correspondences, preference=False):
    """
    Find best indexes of best subset from all correspondences (best sets 'C' from 'M') via maximum
    independent set problem.

    :param preference:                      If preference of bigger sets should be added (max |C|**3)
    :param all_correspondences:             All consistent points list of CorrespondingPoints

    :return:                                Best subset as list of dictionaries with points
                                           [{camera: (x, y), camera2: (x, y), ...}, {...}, ...]
    """
    N = len(all_correspondences)
    model = gp.Model("")
    model.Params.LogToConsole = 0

    used = model.addVars(N, vtype=GRB.BINARY)
    if preference:
        model.setObjective(
            gp.quicksum(used[i]*len(all_correspondences[i].points) ** 3 for i in range(N)), GRB.MAXIMIZE)
    else:
        model.setObjective(
            gp.quicksum(used[i] * len(all_correspondences[i].points) for i in range(N)), GRB.MAXIMIZE)

    # Creating constraints
    for idxA, points_classA in enumerate(all_correspondences):
        for camera, point in points_classA.points.items():
            for idxB, points_classB in enumerate(all_correspondences):
                if idxA == idxB: continue
                if point == points_classB.points.get(camera):
                    model.addConstr(used[idxA] + used[idxB] <= 1)
                    # print("Constraint idx_{} + idx_{} :==: Camera {}  {} vs {}".
                    #       format(idxA, idxB, camera, point, points_classB.points.get(camera)))

    model.optimize()

    used_indexes = list()
    for idx, x in used.items():
        if x.x:
            used_indexes.append(idx)

    return used_indexes


def compute_matrix_M(correspondences, camera1, camera2, threshold_similarity, threshold_distance):
    """
    Compute matrix M for algorithm described in
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf

    :param correspondences:         Pair correspondences between all cameras [dict{camera1: (x,y), camera2: (x,y)}, ..]
    :param camera1:                 Number of first camera int()
    :param camera2:                 Number of second camera int()
    :param threshold_similarity:    Threshold in px for similar distances between correspondences

    :return:                        Symmetrical non-negative matrix M as np.array(N, N)
                                    where N is number of pair correspondences
    """
    def compute_distance(pt1, pt2):
        return np.linalg.norm(np.array([pt1[0], pt1[1]]) - np.array([pt2[0], pt2[1]]))

    n = len(correspondences)
    distances_camera1 = np.zeros((n, n))
    distances_camera2 = np.zeros((n, n))

    # Compute distances between points for each pair of correspondences
    for idx1 in range(0, n-1):
        for idx2 in range(idx1+1, n):
            distances_camera1[idx1][idx2] = distances_camera1[idx2][idx1] = \
                compute_distance(correspondences[idx1].points.get(camera1), correspondences[idx2].points.get(camera1))

            distances_camera2[idx1][idx2] = distances_camera2[idx2][idx1] = \
                compute_distance(correspondences[idx1].points.get(camera2), correspondences[idx2].points.get(camera2))

    # Compute matrix M
    distances = distances_camera1 - distances_camera2
    M = np.zeros((n, n))
    for idx1 in range(0, n - 1):
        for idx2 in range(idx1 + 1, n):
            if distances[idx1][idx2] > threshold_distance or distances[idx1][idx2] < -threshold_distance: continue
            if distances_camera1[idx1][idx2] == 0 or distances_camera2[idx1][idx2] == 0: continue

            max = distances[idx1][idx2] + threshold_similarity
            min = distances[idx1][idx2] - threshold_similarity
            M[idx1][idx2] = M[idx2][idx1] = \
                len(np.where(np.logical_and(min <= distances[:][:], distances[:][:] <= max))[0])

    return M


def leordean_herbert_algorithm(correspondences_pairs, correspondences_more, dist_similarity_threshold=20, dist_threshold=150, conf_threshold=0.14):
    """
    Computation of new pair correspondences based on article
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf.
    From all correspondences compute matrix of distances between correspondences M and its eigen values
    takes as correspondence confidence. Greedy algorithm after takes the most confident consistent pair correspondences.

    :param correspondences_pairs:       Array of correspondences pair points consistent with correspondences_more
                                        (e.g. no duplicate points)
    :param correspondences_more:        Array of more point correspondences points
    :param dist_similarity_threshold:   Threshold for distance for similar distances for computations of matrix M
    :param dist_threshold:              Threshold for points with maximal distance from point to be calculated in matrix M
    :param conf_threshold:              Threshold for minimal confidence score (from eigenvector) of correspondence

    :return:                            New correspondences without duplicate according to Leordean-Herbert alg.
    """

    def compute_solution(eigenvector, correspondences, threshold):
        """
        Implementation of greedy algorithm for finding best non duplicate pair correspondences according to their
        confidence value from principal eigen vector.

        :param eigenvector:             Principal eigen vector of matrix M - confidence
        :param correspondences:
        :param threshold:               Minimal confidence threshold

        :return:
        """
        solution = np.zeros(len(correspondences))
        eigenvector[np.where(eigenvector <= threshold)] = 0

        while not np.all((eigenvector == 0)):
            max_idx = np.argmax(eigenvector)

            if eigenvector[max_idx] == 0:
                return solution

            solution[max_idx] = 1
            eigenvector[max_idx] = 0
            # Remove correspondences in conflict
            best_corr = correspondences[max_idx].points
            for cam, pt in best_corr.items():
                for i, corr in enumerate(correspondences):
                    if pt == corr.points.get(cam):
                        eigenvector[i] = 0

        return solution

    solution = list()
    for camera1 in range(1, 11):
        for camera2 in range(camera1+1, 12):
            # Compute camera matrix M and get its principal eigenvector
            pairs_tmp = [c for c in correspondences_pairs if
                         c.points.get(camera1) is not None and c.points.get(camera2) is not None]
            more_tmp = [c for c in correspondences_more if
                        c.points.get(camera1) is not None and c.points.get(camera2) is not None]
            if len(pairs_tmp) == 0:
                continue

            # Compute positive principal eigenvector of M
            M = compute_matrix_M(pairs_tmp + more_tmp, camera1, camera2, dist_similarity_threshold, dist_threshold)
            w, v = np.linalg.eig(M)
            principal_eigenvector = v[:, np.argmax(w)]
            idx_not_zero = next((i for i, x in enumerate(principal_eigenvector) if x), None)
            principal_eigenvector = principal_eigenvector if principal_eigenvector[idx_not_zero] > 0 else -principal_eigenvector
            # Compute best non duplicate solution for only pair points with confidence greater than threshold
            solution_indices = compute_solution(principal_eigenvector[0:len(pairs_tmp)], pairs_tmp, conf_threshold)
            solution += [pairs_tmp[i] for i in np.where(solution_indices == 1)[0]]

    return solution


def improve_more_correspondences(correspondences, dist_similarity_threshold=25, dist_threshold=300, conf_threshold=0.06):
    """
    Used for improving more points correspondences, based on article from leordean-herbert
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf.
    From all correspondences compute matrix of distances between correspondences M and its eigen values
    takes as correspondence confidence. Take only correspondences with confidence above threshold.

    :param correspondences:             Array of more point correspondences to be improved
    :param dist_similarity_threshold:   Threshold for distance for similar distances for computations of matrix M
    :param dist_threshold:              Threshold for points with maximal distance from point to be calculated in matrix M
    :param conf_threshold:              Threshold for minimal confidence score (from eigenvector) of correspondence

    :return:                            New more point correspondences with confidence above threshold
    """

    for camera1 in range(1, 11):
        for camera2 in range(camera1+1, 12):
            # Get correspondences for given cameras
            corr_tmp = [c for c in correspondences if
                        c.points.get(camera1) is not None and c.points.get(camera2) is not None]
            if len(corr_tmp) == 0:
                continue

            # Compute positive principal eigenvector of M
            M = compute_matrix_M(corr_tmp, camera1, camera2, threshold_similarity=dist_similarity_threshold, threshold_distance=dist_threshold)
            w, v = np.linalg.eig(M)
            principal_eigenvector = v[:, np.argmax(w)]
            idx_not_zero = next((i for i, x in enumerate(principal_eigenvector) if x), None)
            principal_eigenvector = principal_eigenvector if principal_eigenvector[idx_not_zero] > 0 else -principal_eigenvector

            # Remove all correspondences with confidence lower than threshold
            to_remove = [corr_tmp[idx] for idx in np.where(principal_eigenvector <= conf_threshold)[0]]
            for c in to_remove:
                correspondences.remove(c)

    return correspondences

