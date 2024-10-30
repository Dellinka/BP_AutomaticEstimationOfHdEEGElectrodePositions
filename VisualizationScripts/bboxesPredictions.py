"""
Main purpose of this script is visualizing bounding boxes of images via matplotlib library. This script can
visualize predicted boxes from RCNN, detected boxes from template matching and ground truth boxes (if exist). It is also
possible to visualize bounding boxes from more annotations in one image (viz VISUALIZATION_TYPES).

Choose paths to predictions and at the end of script choose specific visualization.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

cameras = [8]
subject = 'TD'
PREDICTIONS_GT = os.path.join('..', 'Dataset', '2d_annotations_csv')
PREDICTIONS_PATH_RCNN = os.path.join('..', 'GeneratedData', 'RCNN')
PREDICTIONS_PATH_TEMPL = os.path.join('..', 'GeneratedData', 'TemplateMatching')
IMAGE_PATH = os.path.join('..', 'Dataset', 'images_dir', subject)

"""
subject = None
PREDICTIONS_GT = None
PREDICTIONS_PATH_RCNN = os.path.join('..', 'RESULTS', 'RCNN')
PREDICTIONS_PATH_TEMPL = os.path.join('..', 'RESULTS', 'TemplateMatching')
IMAGE_PATH = os.path.join('..', 'RESULTS', 'images')
"""


def get_predictions(prediction_file, score):
    """
    This function returns predictions as array of dictionaries

    :param prediction_file:             Path to csv predictions from RCNN etwork
    :param score (float, 0-1)           Min threshold value for confidence score for RCNN predictions or
                                        match value for template matching detections
    :return:                            Array of dictionaries with predictions and labels
                                            [{'bbox'  : bbox parameters as [ymin, xmin, ymax, xmax],
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

            if len(row[3].strip()) == 0:
                row[3] = 1
            elif float(row[3]) < score:
                continue

            predictions.append({'index': row[0],
                                'bbox': np.asarray(np.asarray(row[1].split(), dtype=float), dtype=int),
                                'label': int(row[2]),
                                'score': round(float(row[3]), 2)})

    return predictions


def visualize(img, visualization_type, subject, camera_num, score=0.0, show_score=False, save_f=''):
    """
    Draw predictions into image and plot it

    :param camera_num:
    :param subject:
    :param score:                   Threshold for confidence score or match value
    :param img:                     Image in png format
    :param visualization_type:      Predictions file/files given as VISUALIZATION_TYPES for image - path to predictions
    """

    def draw_predictions(preds, color_1, axis, show_score, black_white=False, color_2='blue'):
        """ Help function for drawing bounding boxes into given image """
        for pred in preds:
            bbox = pred['bbox']
            x, y = bbox[1], bbox[0]
            w, h = bbox[3] - x, bbox[2] - y

            if black_white:
                color = color_1 if pred['label'] == 1 else color_2
                if show_score:
                    axis.text(x, y, pred['score'], style='italic', fontsize=6,
                              bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
            else:
                color = color_1

            if pred['label'] == 'FP':
                color = 'red'

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            axis.add_patch(rect)
            circle = patches.Circle((x + w / 2, y + h / 2), 2, linewidth=1, edgecolor=color, facecolor='none')
            axis.add_patch(circle)
        return axis

    color_a = 'deepskyblue'
    color_b = 'lightcoral'

    # Find out type of prediction/s
    if isinstance(visualization_type, list):
        if subject is None:
            predictions_a = get_predictions(os.path.join(visualization_type[0], 'camera' + str(camera_num) + '.csv'), score)
            predictions_b = get_predictions(os.path.join(visualization_type[1], 'camera' + str(camera_num) + '.csv'), score)
        else:
            predictions_a = get_predictions(
                os.path.join(visualization_type[0], subject, 'camera' + str(camera_num) + '.csv'), score)
            predictions_b = get_predictions(
                os.path.join(visualization_type[1], subject, 'camera' + str(camera_num) + '.csv'), score)
        type_a = list(VISUALIZATION_TYPES.keys())[list(VISUALIZATION_TYPES.values()).index(visualization_type[0])]
        type_b = list(VISUALIZATION_TYPES.keys())[list(VISUALIZATION_TYPES.values()).index(visualization_type[1])]
        type = 'Comparison between predictions'
    else:
        type = type_a = list(VISUALIZATION_TYPES.keys())[list(VISUALIZATION_TYPES.values()).index(visualization_type)]
        if subject is None:
            predictions_a = get_predictions(os.path.join(visualization_type, 'camera' + str(camera_num) + '.csv'), score)
        else:
            predictions_a = get_predictions(
                os.path.join(visualization_type, subject, 'camera' + str(camera_num) + '.csv'), score)
        predictions_b = None

    # Visualize
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(img)
    ax.set_title(type + ' camera ' + str(camera_num))
    # ax.set_title('Custom title camera ' + str(camera_num))

    if predictions_b is None:
        ax = draw_predictions(predictions_a, color_a, ax, show_score, black_white=True, color_2='blue')
        legend = [patches.Patch(color=color_a, label='White sensors'),
                  patches.Patch(color='blue', label='Black sensors')]
        ax.legend(handles=legend, fontsize=10)
    else:
        ax = draw_predictions(predictions_a, color_a, ax, show_score)
        ax = draw_predictions(predictions_b, color_b, ax, show_score)
        legend = [patches.Patch(color=color_a, label=type_a + ' predictions'),
                  patches.Patch(color=color_b, label=type_b + ' predictions')]
        ax.legend(handles=legend, fontsize=10)

    if len(save_f) != 0:
        if not os.path.exists(os.path.dirname(save_f)):
            os.makedirs(os.path.dirname(save_f))

        plt.savefig(save_f)
    else:
        plt.show()


def main(visualization_type):
    """ Main function calling the visualization """
    if visualization_type is None or (type(visualization_type) == list and None in visualization_type):
        print("None in visualization type!")
        return
    for camera_num in cameras:
        img = Image.open(os.path.join(IMAGE_PATH, 'camera' + str(camera_num) + '.png'))
        visualize(img, visualization_type, subject, camera_num, score=0.0, show_score=True)
                  # save_f=os.path.join('..', 'Images', 'RCNN', 'Predictions', subject + '_camera' + str(camera_num) + '.png'))


VISUALIZATION_TYPES = {
    'GT': PREDICTIONS_GT,
    'RCNN': PREDICTIONS_PATH_RCNN,
    'TEMPLATING': PREDICTIONS_PATH_TEMPL,
    'GTvsTEMPL': [PREDICTIONS_GT, PREDICTIONS_PATH_TEMPL],
    'GTvsRCNN': [PREDICTIONS_GT, PREDICTIONS_PATH_RCNN],
    'RCNNvsTEMPL': [PREDICTIONS_PATH_RCNN, PREDICTIONS_PATH_TEMPL]
}

if __name__ == '__main__':
    #main(VISUALIZATION_TYPES['GT'])
    main(VISUALIZATION_TYPES['RCNN'])
    main(VISUALIZATION_TYPES['TEMPLATING'])
    #main(VISUALIZATION_TYPES['GTvsTEMPL'])
    #main(VISUALIZATION_TYPES['GTvsRCNN'])
    #main(VISUALIZATION_TYPES['RCNNvsTEMPL'])

