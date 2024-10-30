import csv
import os
from math import ceil
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np


class Template:
    """
        This class define template images
    """

    def __init__(self, img_path, label, color_positive, color_negative, threshold=0.4):
        """
        :param img_path (string):                   path to template image
        :param label (int):                         label of sensor 0 for black 1 for white
        :param color_positive ((int , int, int)):   color for box visualization for match greater than threshold
        :param color_negative ((int , int, int)):   color for box visualization for match lower than threshold
        :param threshold (float):                   threshold for minimum similarity score
        """

        self.template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.label = label
        self.color_pos = color_positive
        self.color_neg = color_negative
        self.threshold = threshold
        self.height, self.width = self.template.shape


def get_predictions(prediction_file, gt=True, nms_threshold=1., match_threshold=0.):
    """
    This function reads and returns predictions from RCNN network as array of dictionaries.

    :param match_threshold:             Value for template match threshold
    :param nms_threshold:               Value for NMS threshold
    :param gt:                          Boolean specifying if the gt or predictions are in prediction_file
    :param prediction_file:             Path to csv predictions from RCNN network (or gt annotations)
    :return:                            Array of dictionaries with predictions and labels
                                            [{'index' : index of prediction from csv file
                                              'bbox'  : bbox parameters as [ymin, xmin, ymax, xmax],
                                              'label' : 0/1 as black or white sensor}, ... ]
    """
    predictions = []

    with open(prediction_file) as file:
        csv_reader = csv.reader(file, delimiter=';')
        index = 0
        for row in csv_reader:
            if index == 0:
                index += 1
                continue

            # GT annotations
            if gt:
                predictions.append({'index': row[0],
                                    'bbox': np.asarray(row[1].split(), dtype=float),
                                    'label': int(row[2])})
            # Predictions
            elif float(row[3]) > match_threshold:
                predictions.append({'index': row[0],
                                    'bbox': np.asarray(row[1].split(), dtype=float),
                                    'label': int(row[2]),
                                    'match_value': float(row[3])})

            if not gt:
                predictions = non_max_suppression(predictions, nms_threshold)

    return predictions


def get_candidate(box, temp, image):
    """
    Helper function for getting candidate for template matching.
    Candidate is edited cropped predicted image from RCNN so the candidate is
    picture with size same at least as template size + 10 pixel

    :param temp:        template used for template matching
    :param box:         bounding box for predicted sensor
    :return:            information about candidate image
                            {'img' : image used for template matching,
                             'x_min', 'x_max', 'y_min', 'y_max' : as candidate position in !original image!
    """
    # Get bounding box and its parameters from prediction
    y_min, x_min, y_max, x_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = y_max - y_min, x_max - x_min

    h_shift = ceil((max(h, temp.height) + 10 - h) / 2)
    w_shift = ceil((max(w, temp.width) + 10 - w) / 2)
    y_min_new, y_max_new = y_min - h_shift, y_max + h_shift
    if y_min_new < 0:
        y_max_new += abs(y_min_new)
        y_min_new = 0
    elif y_max_new > image.shape[0]:
        y_min_new -= y_max_new - image.shape[0]
        y_max_new = image.shape[0]

    x_min_new, x_max_new = x_min - w_shift, min(x_max + w_shift, image.shape[1])
    if x_min_new < 0:
        y_max_new += abs(x_min_new)
        x_min_new = 0
    elif x_max_new > image.shape[1]:
        y_min_new -= x_max_new - image.shape[1]
        x_max_new = image.shape[1]

    return {'img': image[y_min_new:y_max_new, x_min_new:x_max_new],
            'x_min': x_min_new,
            'x_max': x_max_new,
            'y_min': y_min_new,
            'y_max': y_max_new}


def find_match(image, template, bbox):
    """
    From image creates candidate for sensor detection. Use template matching with given template on candidate
    and return best match as dictionary (more info in this function bellow).

    :param image:                   Image of subject
    :param bbox:                    Predicted bbox from predictions for detecting sensor via template matching
    :param template:                Template used for template matching (viz. class Template)
    :return:                        Best possible match for given template and bbox
                                    **match** is a dictionary viz. code of this function
                                    where bbox is as [ymin, xmin, ymax, xmax]
    """
    # Match template with candidate image
    # returns matching matrix - idx with max value is new predicted top left corner of template
    candidate = get_candidate(bbox, template, image)

    template_matching = cv2.matchTemplate(
        candidate['img'], template.template, cv2.TM_CCOEFF_NORMED   # Can be changed (original TM_CCOEFF_NORMED)
    )
    """
    cv2.imshow('IMG', cv2.resize(candidate['img'], (2*candidate['img'].shape[1], 2*candidate['img'].shape[0])))
    cv2.waitKey(0)
    cv2.imshow('Template', cv2.resize(template.template, (2*template.template.shape[1], 2*template.template.shape[0])))
    cv2.waitKey(0)
    """
    match_locations = np.where(template_matching == np.amax(template_matching))
    x = match_locations[1][0]
    y = match_locations[0][0]

    # print(template_matching[y, x])

    best_match = {
        "bbox": [candidate['y_min'] + y, candidate['x_min'] + x,
                 candidate['y_min'] + y + template.height, candidate['x_min'] + x + template.width],
        "match_value": template_matching[y, x],
        "label": template.label,
        "color": template.color_pos if template_matching[y, x] > template.threshold else template.color_neg,
        "template": template.template
    }

    return best_match


def compute_IoU(a, b, epsilon=1e-5):
    """
    Compute IoU value of two predictions

    :param a, b:                    Predictions/Matches from find_match function (a['bbox'] is array [ymin, xmin, ymax, xmax])
    :param epsilon:                 Small value to prevent division by zero
    :return:
    """
    a_bbox = a['bbox']
    a_xmin, a_ymin = a_bbox[1], a_bbox[0]
    a_xmax, a_ymax = a_bbox[3], a_bbox[2]

    b_bbox = b['bbox']
    b_xmin, b_ymin = b_bbox[1], b_bbox[0]
    b_xmax, b_ymax = b_bbox[3], b_bbox[2]

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a_xmin, b_xmin)
    y1 = max(a_ymin, b_ymin)
    x2 = min(a_xmax, b_xmax)
    y2 = min(a_ymax, b_ymax)

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a_xmax - a_xmin) * (a_ymax - a_ymin)
    area_b = (b_xmax - b_xmin) * (b_ymax - b_ymin)
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def non_max_suppression(objects, nms_threshold=1.):
    """
    Filter objects overlapping with IoU over threshold by keeping only the one with maximum score.

    :param objects:                         List of detections/matches from find_match function
    :param nms_threshold:                   Threshold for NMS -> predictions with IoU >= mns_threshold will be suppressed
    :return:                                Suppressed detections as array of dictionaries
    """
    sorted_objects = sorted(objects, key=lambda obj: obj['match_value'], reverse=True)
    filtered_objects = []
    for object_ in sorted_objects:
        overlap_found = False
        for filtered_object in filtered_objects:
            iou = compute_IoU(object_, filtered_object)
            if iou > nms_threshold:
                overlap_found = True
                break
        if not overlap_found:
            filtered_objects.append(object_)
    return filtered_objects


def load_templates():
    """
    This function creates Template objects for template matching from images in
    ./template/ directory.

    :return:            python array of Template objects
    """
    template_path = os.path.dirname(os.path.abspath(__file__)) + '/templates/'
    templates_files = [f for f in listdir(template_path)
                       if isfile(join(template_path, f))]
    templates = []
    for file in templates_files:
        label = 0 if file.split('_')[0] == 'BLACK' else 1
        templates.append(Template(template_path + file,
                                  label=label,
                                  color_positive=(255, 255, 255) if label == 0 else (0, 0, 0),
                                  color_negative=(255, 0, 0) if label == 0 else (255, 0, 0)))

    return templates
