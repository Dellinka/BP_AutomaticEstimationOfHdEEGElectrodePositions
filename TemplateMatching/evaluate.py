import os

from main import fit_templates_and_save, get_predictions

import numpy as np
import scipy.optimize

GT_BBOXES_PATH = os.path.join('..', 'Dataset', '2d_annotations_csv')
RCNN_BBOXES_PATH = os.path.join('..', 'GeneratedData', 'RCNN')
TEMPLATING_BBOXES_PATH = os.path.join('..', 'GeneratedData', 'TemplateMatching')
subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


class Precision:
    """
    Class for computing precision of detection for one subject through all 11 cameras.
    """

    def __init__(self, predictions_path, subject, nms_threshold, match_threshold):
        self.path = predictions_path
        self.subject = subject
        self.gt_centers = list()
        self.pred_centers = list()
        self.gt_annotations = list()
        self.pred_annotations = list()
        self.nms_threshold = nms_threshold
        self.match_threshold = match_threshold

        for cam_num in range(1, 12):
            gt_annotations = get_predictions(os.path.join(GT_BBOXES_PATH, subject, 'camera' + str(cam_num) + '.csv'))
            pred_annotations = get_predictions(os.path.join(predictions_path, subject, 'camera' + str(cam_num) + '.csv'),
                                               gt=False, nms_threshold=nms_threshold,
                                               match_threshold=match_threshold)
            self.gt_annotations.append(gt_annotations)
            self.pred_annotations.append(pred_annotations)

            # Create array of arrays with ground truth centers
            centers = list()
            for gt_annot in gt_annotations:
                centers.append(self.compute_center(gt_annot['bbox']))
            self.gt_centers.append(centers)

            # Create array of arrays with predicted centers
            centers = list()
            for pred_annot in pred_annotations:
                centers.append(self.compute_center(pred_annot['bbox']))
            self.pred_centers.append(centers)

        self.gt_centers = np.asarray(self.gt_centers)
        self.pred_centers = np.asarray(self.pred_centers)

    @staticmethod
    def compute_center(bbox):
        """
        Compute center of bounding box

        :param bbox:        Parameters of bbox from annotation as (ymin, xmin, ymax, xmax)
        :return:            Coordinates of center as numpy array (x, y)
        """
        x = (bbox[1] + bbox[3]) / 2
        y = (bbox[0] + bbox[2]) / 2
        return np.array([x, y])

    def distance_from_center(self):
        """
        Computes mean and median distances (using numpy.linalg.norm) between
        predicted and ground truth points center.
        """
        distances = list()  # Mean distance from gt center through cameras

        # Iteration through cameras
        for idx, pred_centers_array in enumerate(self.pred_centers):
            gt_centers_array = np.asarray(self.gt_centers[idx])

            # Iteration through predictions from camera num. idx
            for pred_center in pred_centers_array:
                center_as_array = np.ones((len(gt_centers_array), 2)) * pred_center
                dist = np.linalg.norm(gt_centers_array - center_as_array, axis=1)
                distances.append(np.amin(dist))

        return {'mean center distance': np.mean(distances), 'median center distance': np.median(distances)}

    def compute_FP(self):
        """
        Compute number of False positive predictions. False positive in this code is defined as a prediction
        which distance to nearest ground truth center is more than 10 pixels.

        :return: Average number of False positive predictions among the data
        """
        FP_lst = list()  # Number of false positive predictions through cameras

        # Iteration through cameras
        for idx, pred_centers_array in enumerate(self.pred_centers):
            gt_centers_array = np.asarray(self.gt_centers[idx])
            FP = 0

            # Iteration through predictions from camera num. idx
            for pred_center in pred_centers_array:
                center_as_array = np.ones((len(gt_centers_array), 2)) * pred_center
                dist = np.linalg.norm(gt_centers_array - center_as_array, axis=1)
                if np.amin(dist) > 10: FP += 1
            FP_lst.append(FP)

        # print("{} FP: {}".format(self.subject, FP_lst))
        return {'FP list': FP_lst, 'average FP': np.average(FP_lst)}

    def compute_FN(self):
        """
        Compute number of False negative predictions. False negative in this code is defined as ground truth box
        which distance to closest predicted center is higher than 10 pixels.

        :return: Average number of False negative predictions among the data
        """
        FN_lst = list()  # Number of false negative predictions through cameras

        # Iteration through cameras
        for idx, gt_centers_array in enumerate(self.gt_centers):
            pred_centers_array = np.asarray(self.pred_centers[idx])
            FN = 0

            # Iteration through ground truth centers from camera num. idx
            for gt_center in gt_centers_array:
                center_as_array = np.ones((len(pred_centers_array), 2)) * gt_center
                dist = np.linalg.norm(pred_centers_array - center_as_array, axis=1)
                if np.amin(dist) > 10: FN += 1
            FN_lst.append(FN)

        # print("{} FN: {}".format(self.subject, FN_lst))
        return {'FN list': FN_lst, 'average FN': np.average(FN_lst)}

    def compute_misclassified(self):
        """
        Compute average number of misclassified classes in predictions.

        :return: average number of misclassified classes
        """
        misclassified = list()  # Number of misclassified predictions through cameras

        # Iteration through cameras
        for camera_idx, pred_centers_array in enumerate(self.pred_centers):
            gt_centers_array = np.asarray(self.gt_centers[camera_idx])
            miss = 0

            # Iteration through predictions from camera num. idx
            for prediction_idx, pred_center in enumerate(pred_centers_array):
                center_as_array = np.ones((len(gt_centers_array), 2)) * pred_center
                distance = np.linalg.norm(gt_centers_array - center_as_array, axis=1)
                min_dist = np.amin(distance)
                min_dist_idx = np.argmin(distance)
                if min_dist <= 10 and self.gt_annotations[camera_idx][min_dist_idx]['label'] != \
                        self.pred_annotations[camera_idx][prediction_idx]['label']:
                    miss += 1
            misclassified.append(miss)

        # print("{} misclassified: {}".format(self.subject, misclassified))
        return {'misclassified list': misclassified, 'average misclassified': np.average(misclassified)}


def eval_all(path=TEMPLATING_BBOXES_PATH, nms=1., score=0.):
    """
    Compute everything for all subjects and cameras.

    :param path:                path to predictions from RCNN or Template Matching
    :param nms:                 Threshold for Non maximal suppression
    :param score:               Threshold for score (confidence score for RCNN or match value for Template Matching)
    :return:                    Mean, median, FP, FN and misclassified values in dictionary
    """
    mean, median = list(), list()
    FP, FN = list(), list()
    miss = list()

    for s in subjects:
        detections = Precision(path, s, nms_threshold=nms, match_threshold=score)
        distance = detections.distance_from_center()
        mean.append(distance['mean center distance'])
        median.append(distance['median center distance'])
        FP.append(detections.compute_FP()['average FP'])
        FN.append(detections.compute_FN()['average FN'])
        miss.append(detections.compute_misclassified()['average misclassified'])

    return {'mean': np.mean(np.asarray(mean)), 'median': np.mean(np.asarray(median)),
            'FP': np.average(FP), 'FN': np.average(FN), 'misclassified': np.average(miss)}


def match_all(nms=0.0541, score=0.45418):
    """
    Match all predictions (all subjects) from RCNN for each subject from dataset
    -> create predictions using template matching from RCNN predictions.

    :params nms             Non maximal suppression threshold used in template matching
    :params score           Threshold for match value from template matching
    """
    path_imgs = os.path.join('..', 'Dataset', 'images_dir')
    output = os.path.join('..', 'GeneratedData', 'TemplateMatching')
    predictions = os.path.join('..', 'GeneratedData', 'RCNN')

    for s in subjects:
        fit_templates_and_save(images_dir=os.path.join(path_imgs, s),
                               predictions_dir=os.path.join(predictions, s),
                               output_dir=os.path.join(output, s),
                               nms=nms, score=score)
        print("Subject {} DONE".format(s))


def get_optimal_params():
    """
    Computation of optimal nms and score threshold using the scipy.optimize.fminbound function.
    WARNING: This computation takes A LOT OF time!

    Best for our RCNN and dataset - Score 0.4769332332985249, OPTIMAL nms 0.05502587930118583, mean 2.944, FN 8.9886, FP 2.7727
    """
    params = {}

    def optimal_nms(score_fixed):
        """
        Find optimal nms (nms_opt_tmp) threshold for fixed confidence score such as
        the average number of FN is lower than 9 (for CCOEFF matching)
        """

        def evaluate(nms, score):
            """
            Evaluate network result on fixed nms and score threshold.
            """
            data_ = eval_all(nms=nms, score=score)
            mean = data_['mean'] if data_['FN'] < 9 else data_['mean'] * data_['FN']
            print('Score {}, nms {}, mean {}, FN {}, FP {}, miss {}'.
                  format(score, nms, mean, data_['FN'], data_['FP'], data_['misclassified']))
            return mean

        # Optimal nms for fixed score
        nms_opt_tmp = scipy.optimize.fminbound(lambda nms: evaluate(nms, score_fixed), 0, 0.3, xtol=0.01)

        data = eval_all(nms=nms_opt_tmp, score=score_fixed)
        mean = data['mean'] if data['FN'] < 9 else data['mean'] * data['FN']

        print('------------------------')
        print('Score {}, OPTIMAL nms {}, mean {}, FN {}, FP {}, miss {}'.
              format(score_fixed, nms_opt_tmp, mean, data['FN'], data['FP'], data['misclassified']))
        print('------------------------')
        params[(score_fixed, nms_opt_tmp)] = mean
        return mean

    scipy.optimize.fminbound(lambda score: optimal_nms(score), 0, 1, xtol=0.01)
    (score_opt, nms_opt) = max(params, key=params.get)
    nms_opt = round(nms_opt, 5)
    score_opt = round(score_opt, 5)

    return {'nms': nms_opt, 'score': score_opt}


if __name__ == '__main__':
    print("Evaluation of all predictions in {} path \n"
          "You can generate new predictions uncommenting the match_all() function\n"
          "in evaluate.py file".format(TEMPLATING_BBOXES_PATH))
    # params = get_optimal_params()

    # TM_CCOEFF_NORMED  0.05503, 0.47693 ; RCNN labels 0.0541,  0.45418
    # TM_CCORR_NORMED   0.01033, 0.76873 ; RCNN labels 0.03738, 0.88079

    # match_all(nms=0.0541, score=0.45418)
    result = eval_all(TEMPLATING_BBOXES_PATH, nms=0.0541, score=0.45418)
    print("Mean of center distances through subjects: ", np.round(result['mean'], 4))
    print("Median of center distances through subjects: ", np.round(result['median'], 4))
    print("Average number of false positive predictions: ", np.round(result['FP'], 4))
    print("Average number of false negative predictions: ", np.round(result['FN'], 4))
    print("Average number of misclassified predictions: ", np.round(result['misclassified'], 4))
