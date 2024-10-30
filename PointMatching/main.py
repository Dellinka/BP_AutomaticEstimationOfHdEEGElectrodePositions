"""
In this directory is implemented code for finding corresponding points between predictions and all cameras.
Main idea of the code is described bellow :

1)  Finding all possibly corresponding points between cameras A and B. For each predicted point from camera A find
    set 'Y' of possibly corresponding points from camera B (as points lying on or near the epipolar line in image B).
2)  For each possibly corresponding pair compute 3D coordinate and re-project it to all camera images.
3)  From reprojected points select only the consistent ones (same color and reprojection error < epsilon) which creates
    the set 'C' of all possibly corresponding points through all cameras.
4)  Let all sets 'C' call set 'M'. Final corresponding points will be found as the best subset of 'M' (f.e. with greedy algorithm).
    - This algorithm was improved using the Leonard herbert algorithm https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf.
        - First only 3+ points correspondences are computed (using Gurobi library)
        - After the consistent pair correspondences were computed
        - Using the leonard Herbert algorithm all final correspondences were computed

Note: All sets from which to find the best ones are saved in global variable ALL_CORRESPONDENCES. It is list of
dictionaries (sets 'C' - possibly corresponding points) and represents the set 'M'.
"""

import os
import numpy as np
from CorrespondingPoints import CorrespondingPoints

from computationFunctions_PM import find_possible_correspondences, linear_triangulation, compute_reprojected_point, \
    find_consistent_from_reprojected, find_best_subset_greedy, find_best_subset_indexes_gurobi, find_common_subsets, \
    compute_fundamental_matrix, leordean_herbert_algorithm, improve_more_correspondences
from helpFunctions import read_predictions, read_camera_constraints, save_correspondences
from visualization_PM import visualize_epipolar_lines

# Global paths
IMG_DATA_PATH = os.path.join('..', 'Dataset', 'images_dir')


def create_and_save_correspondences(predictions_dir, camera_matrices, camera_constraint, output_dir,
                                    epsilon1=8, epsilon2=5,
                                    dist_similarity_threshold_more=25, dist_threshold_more=300, conf_threshold_more=0.06,
                                    dist_similarity_threshold_pairs=20, dist_threshold_pairs=150, conf_threshold_pairs=0.14):
    """
    In this function is implemented the algorithm described at the beginning of this file.


    :param predictions_dir           Path to directory with predictions from template matching
    :param camera_matrices           Path to camera matrices stored in Config
    :param camera_constraint         Path to camera constraints stored in Config
    :param output_dir                Path to output directory
    :param epsilon1                  Threshold distance for finding possibly corresponding points (max distance from epipolar line)
    :param epsilon2                  Threshold distance for finding corresponding point from reprojected 3d points
    :params other                    Parameters used in leonard herbert based algorithm for finding final correspondences
    """
    # Global help variables
    all_predictions = {}
    all_camera_matrices = {}
    two_point_correspondences = list()
    three_point_correspondences = list()

    # Read predictions from all cameras
    for camera_num in range(1, 12):
        all_predictions[camera_num] = read_predictions(
            os.path.join(predictions_dir, 'camera' + str(camera_num) + '.csv'))
        all_camera_matrices[camera_num] = np.loadtxt(
            os.path.join(camera_matrices, 'camera' + str(camera_num) + '.out'))

    # Read camera constraints
    camera_constraints = read_camera_constraints(camera_constraint)

    # 1) Go through all possible pairs of cameras according to camera constraints (from Config directory)
    for cameraA, other_cams in camera_constraints.items():
        for cameraB in other_cams:
            if cameraB <= cameraA: continue
            # Find possible correspondences between predictions in two given cameras (documentation => set 'Y')
            possibly_corresponding_pairs = find_possible_correspondences(all_predictions[cameraA],
                                                                         all_predictions[cameraB],
                                                                         all_camera_matrices[cameraA],
                                                                         all_camera_matrices[cameraB],
                                                                         epsilon=epsilon1)

            # 2) For each possibly corresponding pair compute 3D coordinate -> re-project it to other images ->
            # -> check if reprojected points are consistent
            for pointA, correspondingPts in possibly_corresponding_pairs.items():
                color = pointA[1]
                pointA = pointA[0]

                """
                # Visualization of epipolar line for pointA and corresponding points in imageB
                if cameraA==5 and len(correspondingPts) > 0:
                    visualize_epipolar_lines(os.path.join(IMG_DATA_PATH, 'AH', 'camera' + str(cameraA) + '.png'),
                                             os.path.join(IMG_DATA_PATH, 'AH', 'camera' + str(cameraB) + '.png'),
                                             compute_fundamental_matrix(all_camera_matrices[cameraA], all_camera_matrices[cameraB]),
                                             p1=np.asarray(pointA).reshape((2, 1)),
                                             p2=np.transpose(np.asarray(correspondingPts)))
                                             # save_f=os.path.join('..', 'Images', 'Corresponding', 'epipolarLine.png'))
                """

                for pointB in correspondingPts:
                    # Compute 3D point (np.array(3,))
                    coord_3d = linear_triangulation(np.asarray(pointA),
                                                    np.asarray(pointB),
                                                    all_camera_matrices[cameraA],
                                                    all_camera_matrices[cameraB])

                    # Compute reprojected point from 3D coordinate into all images
                    # 3) Check if it is consistent (color, reprojection error < epsilon) with pointA and pointB
                    consistent = {}
                    common_AB = camera_constraints[cameraA].intersection(camera_constraints[cameraB])
                    for cameraC in common_AB:
                        cameraMatrixC = all_camera_matrices[cameraC]
                        pointC = compute_reprojected_point(coord_3d, cameraMatrixC)

                        consistent_point = find_consistent_from_reprojected(np.asarray(pointC), color, all_predictions[cameraC], epsilon=epsilon2)
                        if consistent_point is not None:
                            consistent[cameraC] = [consistent_point, pointC, coord_3d]

                    # Create two point correspondences
                    corresponding_points = CorrespondingPoints(color)
                    corresponding_points.points[cameraA] = pointA
                    corresponding_points.points[cameraB] = pointB
                    corresponding_points.coord_3d = coord_3d
                    two_point_correspondences.append(corresponding_points)

                    # Add correspondences with more than 2 points
                    common = find_common_subsets(set(consistent.keys()), camera_constraints)
                    for cameras in common:
                        corresponding_points = CorrespondingPoints(color)
                        corresponding_points.points[cameraA] = pointA
                        corresponding_points.points[cameraB] = pointB
                        for cameraC in cameras:
                            corresponding_points.points[cameraC] = consistent[cameraC][0]
                            corresponding_points.points_reprojected[cameraC] = consistent[cameraC][1]
                            corresponding_points.coord_3d = consistent[cameraC][2]
                        three_point_correspondences.append(corresponding_points)

    # print("3-points before gurobi: " + str(len(three_point_correspondences)))
    # 4) Find the best subsets S from set M -> ALL_CORRESPONDENCES
    # final_correspondences = find_best_subset_greedy(two_point_correspondences + three_point_correspondences)
    used_indexes = find_best_subset_indexes_gurobi(three_point_correspondences, preference=False)
    opt_more_correspondences = [three_point_correspondences[idx] for idx in used_indexes]
    # print("3-points after gurobi: " + str(len(opt_more_correspondences)))
    opt_more_correspondences = improve_more_correspondences(opt_more_correspondences,
                                                            dist_similarity_threshold=dist_similarity_threshold_more,
                                                            dist_threshold=dist_threshold_more,
                                                            conf_threshold=conf_threshold_more)
    # print("3-points after LH: " + str(len(opt_more_correspondences)))

    # print("2-points all: " + str(len(two_point_correspondences)))
    # Compute pair correspondences consistent with 3+ correspondences
    for correspondence in opt_more_correspondences:
        if len(correspondence.points) == 2: continue
        for camera, point in correspondence.points.items():
            for idx, corresponding in enumerate(two_point_correspondences[:]):
                if point == corresponding.points.get(camera):
                    two_point_correspondences.remove(corresponding)
    # save_correspondences(two_point_correspondences, os.path.join(output_dir, 'pairs_beforeLH.csv'))
    # print("2-points consistent: " + str(len(two_point_correspondences)))

    # 5) Leordean-Herbert method - finding best pair correspondences according to distances of correspondences through images
    possibly_opt_pairs = leordean_herbert_algorithm(two_point_correspondences, opt_more_correspondences,
                                                    dist_similarity_threshold=dist_similarity_threshold_pairs,
                                                    dist_threshold=dist_threshold_pairs,
                                                    conf_threshold=conf_threshold_pairs)
    # print("2-points after LH: " + str(len(possibly_opt_pairs)))

    # save_correspondences(possibly_opt_pairs, os.path.join(kwargs['output_dir'], 'improved_pairs.csv'))
    used_indexes = find_best_subset_indexes_gurobi(possibly_opt_pairs)
    opt_pairs = [possibly_opt_pairs[idx] for idx in used_indexes]

    # print("2-points after Gurobi: " + str(len(opt_pairs)))

    final_correspondences = opt_more_correspondences + opt_pairs

    # save_correspondences(opt_pairs, os.path.join(output_dir, 'opt_pairs.csv'))
    save_correspondences(final_correspondences, os.path.join(output_dir, 'corresponding_points.csv'))
