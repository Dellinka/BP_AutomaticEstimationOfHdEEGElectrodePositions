"""
This is evaluating script for template matching. Aims to make more precise predictions from RCNN network
via template matching algorithm.
"""

import os
import cv2
from templateMatching import load_templates, get_predictions, find_match, non_max_suppression


def save_detections(detections, output_file):
    """
    This is help function for saving predicted sensors.

    :params detections:             Detections computed in fit_templates_and_save
    :params output_file:            File where to store computed sensors
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    f = open(output_file, 'w')
    f.write('Index; Bbox ymin xmin ymax xmax; Label; Match value')
    for idx in range(len(detections)):
        detection = detections[idx]

        f.write('\n' + str(detection['index']) + '; '
                + str(detection['bbox'][0]) + ' '
                + str(detection['bbox'][1]) + ' '
                + str(detection['bbox'][2]) + ' '
                + str(detection['bbox'][3]) + ' ' + '; '
                + str(detection['label']) + '; '
                + str(detection['match_value']))
    f.close()


def fit_templates_and_save(images_dir, predictions_dir, output_dir, score=0.0541, nms=0.45418):
    """
    This function runs the template matching algorithm. Improve predictions from RCNN
    and stores them in specified output file.

    :param images_dir               Directory with original 11 images in png format (images_dir="../result/images/")
    :param predictions_dir          Directory with predictions from RCNN network (predictions_dir="../result/RCNN/" )
    :param output_dir               Directory for output predictions (output_dir="../result/TemplateMatching/")
    :param nms (optional)           Threshold for non maximal suppression
    :param score (optional)         Threshold for match value from template matching
    """
    templates = load_templates()

    for idx in range(1, 12):    # Loop through each camera
        subject_img = cv2.imread(os.path.join(images_dir, 'camera' + str(idx) + '.png'), cv2.IMREAD_GRAYSCALE)
        predictions = get_predictions(os.path.join(predictions_dir, 'camera' + str(idx) + '.csv'))
        output_file = os.path.join(output_dir, 'camera' + str(idx) + '.csv')

        detections = []
        for prediction in predictions:  # Loop through all predictions
            best_detection = None

            for template in templates:      # Loop through all templates
                if template.label == prediction['label']:
                    match = find_match(subject_img, template, prediction['bbox'])
                    if best_detection is None or best_detection['match_value'] < match['match_value']:
                        best_detection = match

            best_detection['index'] = prediction['index']
            if best_detection['match_value'] >= score:
                detections.append(best_detection)

        detections = non_max_suppression(detections, nms_threshold=nms)
        save_detections(detections, output_file)
