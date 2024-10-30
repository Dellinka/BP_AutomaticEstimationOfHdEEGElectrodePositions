import csv
import os
import numpy as np

from helpFunctions import read_correspondences
from main import create_and_save_correspondences

correspondences_dir = os.path.join('..', 'TEST_correspondences', 'Gurobi_all')  # Originally ('..', 'GeneratedData', 'Correspondences')
blackSensors = ['FidNz', 'FidT9', 'FidT10', 'E31', 'E36', 'E67', 'E72', 'E111', 'E114', 'E119',
                'E168', 'E173', 'E199', 'E219', 'E224', 'E234', 'E237', 'E244', 'E247', 'E257']
subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


def save_all_correspondences(eps1=8, eps2=5, similarity_more=25, dist_more=300, conf_more=0.06,
                             similarity_pairs=20, dist_pairs=150, conf_pairs=0.14):
    """
    This function calls computation of all correspondences and ave them in  GeneratedData directory.
    All parameters were used only for optimal thresholds computation.
    """
    camera_matrices_dir = os.path.join('..', 'Config', 'CameraMatrices')
    camera_constraints = os.path.join('..', 'Config', 'cameraConstraints.txt')

    for idx, subject in enumerate(subjects):
        predictions_dir = os.path.join('..', 'GeneratedData', 'TemplateMatching', subject)
        output_dir = os.path.join('..', 'GeneratedData', 'Correspondences')    # Originally ('..', 'GeneratedData', 'Correspondences', subject)
        create_and_save_correspondences(predictions_dir=predictions_dir,
                                        camera_matrices=camera_matrices_dir,
                                        camera_constraint=camera_constraints,
                                        output_dir=output_dir)

        print("{} out of {} done".format(idx+1, len(subjects)))

    # print("Two point {} {} ".format(np.average(np.asarray(two_all)), np.asarray(two_all)))
    # print("Three+ point {} {}".format(np.average(np.asarray(three_all)), np.asarray(three_all)))
    # print("Two point after removing three+ {} {}".format(np.average(np.asarray(twoNoThree_all)), np.asarray(twoNoThree_all)))


def eval_correspondence_3d(subject):
    """
    Evaluate predicted correspondences using 3d points.

    :param subject:             Name of subject to be evaluated

    Returns:                    Number of predicted correspondences and average distance between 3d points

    """
    def load_3d(file):
        """Reads and returns 3d original corresponding points from Dataset/3d_annotations_csv_orig"""
        coords_3d = list()
        labels = list()
        with open(file) as f:
            csv_reader = csv.reader(f, delimiter=';')
            for row in csv_reader:
                coords_3d.append(np.array([float(row[1]), float(row[2]), float(row[3])]))
                labels.append(row[0])

        return coords_3d, labels

    distances = list()

    # Load data
    correspondences_file = os.path.join(correspondences_dir,  subject, 'corresponding_points.csv')
    coords_3d_file = os.path.join('..', 'Dataset', '3d_annotations_csv_orig', subject + '.csv')

    coords_3d_correspondences = [(p.coord_3d, p.color) for p in read_correspondences(correspondences_file)]
    coords_3d_original, orig_labels = load_3d(coords_3d_file)

    points_num = len(coords_3d_correspondences)
    orig_black = [coords_3d_original[idx] for idx in np.where([1 if c in blackSensors else 0 for c in orig_labels])[0]]
    orig_white = [coords_3d_original[idx] for idx in np.where([0 if c in blackSensors else 1 for c in orig_labels])[0]]

    # Find closest original
    for idx, correspondence in enumerate(coords_3d_correspondences):
        point, color = correspondence[0], correspondence[1]

        if color == 0:  # BLACK
            all_distances = np.array([np.linalg.norm(point - orig) for orig in orig_black])
        else:
            all_distances = np.array([np.linalg.norm(point - orig) for orig in orig_white])

        closest_idx = int(np.argmin(all_distances))
        # print(str(idx) + ' ' + str(distances[closest_idx]) + '    -    ' + str(correspondence) + " vs " + str(
        #    coords_3d_original[closest_idx]))
        distances.append(all_distances[closest_idx])

    return np.average(distances), points_num


def eval_correspondence_2d(subject):
    """
    Evaluate predicted correspondences using 2d points.

    :param subject:     Name of subject to be evaluated

    Returns:            Indexes of right, wrong and unknown predicted correspondences
    """
    def load_2d():
        """
        Reads and returns 2D original correspondences from Dataset/2d_annotations_csv_orig
        as dictionary of array of corresponding points for each camera
        {1: [('camera' : [x, y]), ...] , ...}.
        """
        coords = {}

        for cam in range(1, 12):
            coords[cam] = list()
            file = os.path.join('..', 'Dataset', '2d_annotations_csv_orig', subject, 'camera' + str(cam) + '.csv')
            with open(file) as f:
                csv_reader = csv.reader(f, delimiter=';')
                first = True
                for row in csv_reader:
                    if first:
                        first = False
                        continue

                    coords[cam].append((row[0], np.array([float(row[1]), float(row[2])])))

        return coords

    correspondences_file = os.path.join(correspondences_dir, subject, 'corresponding_points.csv')
    coords_2d_correspondences = [p.points for p in read_correspondences(correspondences_file)]
    coords_2d_original = load_2d()

    notknown = list()
    ok = list()
    error = list()

    for idx, correspondence in enumerate(coords_2d_correspondences):
        labels = list()
        for cam, point in correspondence.items():
            original = np.array([p[1] for p in coords_2d_original[cam]])
            all_distances = np.array([np.linalg.norm(np.asarray(point) - orig) for orig in original])
            closest_idx = int(np.argmin(all_distances))
            if all_distances[closest_idx] <= 15:
                labels.append(coords_2d_original[cam][closest_idx][0])

        if len(labels) < 2:
            notknown.append(idx)
            # print(idx, " UNKNOWN")
        elif len(set(labels)) == 1:
            ok.append(idx)
            # print(idx, ' OK')
        else:
            error.append(idx)
            # print(idx, ' WRONG')

    return notknown, ok, error


def eval_all():
    """
    Evaluate predicted correspondences using eval_correspondence_2d (right, wrong, unknown correspondences)
    and eval_correspondence_3d (number of predicted points, distances in 3d).
    """
    print("Evaluating: " + str(correspondences_dir))
    unknown, right, wrong = list(), list(), list()

    for s in subjects:
        u, r, w = eval_correspondence_2d(s)
        unknown.append(len(u))
        right.append(len(r))
        wrong.append(len(w))

    print('Average number of right: {}, wrong: {} and unknown: {} correspondences'.
          format(np.average(np.asarray(right)), np.average(np.asarray(wrong)), np.average(np.asarray(unknown))))

    distances_3d = {}
    all_points = {}
    for s in subjects:
        pts, all_num = eval_correspondence_3d(s)
        distances_3d[s] = pts
        all_points[s] = all_num

    # print('Average distances: ' + str(distances_3d))
    # print('Number of predicted correspondences: ' + str(all_points))
    print('Average number of predicted correspondences: ' +
          str(np.average(np.asarray(list(all_points.values())))))
    print('Average of average distances: ' +
          str(round(np.average(np.asarray(list(distances_3d.values()))), 4)))
    print("----------------------------------------------------------------")

    return all_points, distances_3d, right, wrong, unknown


def opt_epsilons():
    print("WARNING! Add epsilons in find_all_correspondences!!")
    f = open('file_optimal_epsilons.csv', 'w')
    f.write('epsilon1; epsilon2; Number of correspondences; 3D distances; Right; Wrong; Unknown')

    for epsilon1 in range(5, 11):
        for epsilon2 in range(5, 11):
            save_all_correspondences(eps1=epsilon1, eps2=epsilon2)
            print("Epsilon 1: {}, epsilon2: {}".format(epsilon1, epsilon2))
            all_points, distances_3d, right, wrong, unknown = eval_all()

            line = '\n' + str(epsilon1) + '; ' + str(epsilon2) + '; ' \
                   + str(np.average(np.asarray(list(all_points.values())))) + '; ' \
                   + str(round(np.average(np.asarray(list(distances_3d.values()))), 4)) + '; ' \
                   + str(np.average(np.asarray(right))) + '; ' \
                   + str(np.average(np.asarray(wrong))) + '; ' \
                   + str(np.average(np.asarray(unknown))) + '; '

            f.write(line)
    f.close()


def opt_thresholds():
    """
    Bruteforce computation of thresholds for correspondence finding algorithm used
    in Leonard herbert algorithm. Optimal was then manually chosen.
    """

    print("WARNING! Add thresholds in find_all_correspondences!!")
    f = open('file_all_optimal_thresholds.csv', 'w')
    f.write('Threshold for close points; Threshold for similar distance; Confidence threshold; Number of correspondences; 3D distances; Right; Wrong; Unknown')

    for distance in range(150, 525, 75):
        for similarity in range(15, 30, 5):
            for conf in range(8, 20, 2):
                conf /= 100
                save_all_correspondences(similarity_pairs=similarity, dist_pairs=distance, conf_pairs=conf)
                print("Distance: {}, Similarity: {}, Confidence: {}".format(distance, similarity, conf))
                all_points, distances_3d, right, wrong, unknown = eval_all()

                line = '\n' + str(distance) + '; ' + str(similarity) + '; ' + str(conf) + ';'\
                       + str(np.average(np.asarray(list(all_points.values())))) + '; ' \
                       + str(round(np.average(np.asarray(list(distances_3d.values()))), 4)) + '; ' \
                       + str(np.average(np.asarray(right))) + '; ' \
                       + str(np.average(np.asarray(wrong))) + '; ' \
                       + str(np.average(np.asarray(unknown))) + '; '

                f.write(line)

                if np.average(np.asarray(list(all_points.values()))) < 220:
                    break
    f.close()


if __name__ == '__main__':
    # save_all_correspondences()
    eval_all()


