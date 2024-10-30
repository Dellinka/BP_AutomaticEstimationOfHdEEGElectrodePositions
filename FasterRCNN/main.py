"""
Final script of RCNN network used in run.py for prediction of bounding boxes for sensors.
Function predict and save takes directory to 11 images in png format and predict sensors
via trained neural network.

This directory is based on implementation of Faster RCNN from https://github.com/chenyuntc/simple-faster-rcnn-pytorch
"""
import os

import torch

from data.util import read_image
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer


def predict_and_save(**kwargs):
    """
    Predict bounding boxes for sensors for images in given directory and save it into given output directory
    as csv files.

    :param kwargs:
        ** input_dir:                 Directory with 11 png images used for prediction (.png extension)
        ** output_dir:                Directory for output files with predictions in CSV format (.csv extension)
    """
    # Load pretrained network
    faster_rcnn = FasterRCNNVGG16(n_fg_class=2, anchor_scales=[4, 8, 16, 32], ratios=[0.8, 1, 1.25])
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'flipped2.zip'))
    trainer.eval()

    opt_nms = 0.12192              # Non-maximal suppression threshold for prediction
    opt_score = 0.20665            # Confidence score threshold for prediction

    for camera_idx in range(1, 12):
        img_path = os.path.join(kwargs['input_dir'], 'camera' + str(camera_idx) + '.png')
        pred_path = os.path.join(kwargs['output_dir'], 'camera' + str(camera_idx) + '.csv')

        img = read_image(img_path)
        img = torch.from_numpy(img)[None]

        bboxes, labels, scores = trainer.faster_rcnn.predict(img, visualize=True, use_preset=False,
                                                             nms=opt_nms, score=opt_score)

        if not os.path.exists(os.path.dirname(pred_path)):
            os.makedirs(os.path.dirname(pred_path))

        f = open(pred_path, 'w')
        f.write('Index; Bbox ymin xmin ymax xmax; Label; Confidence score;')
        for idx in range(len(labels[0])):
            f.write('\n' + str(idx) + '; '
                    + str(bboxes[0][idx][0]) + ' ' + str(bboxes[0][idx][1]) + ' '
                    + str(bboxes[0][idx][2]) + ' ' + str(bboxes[0][idx][3]) + ' ' + '; '
                    + str(labels[0][idx]) + '; '
                    + str(round(scores[0][idx], 2)))
        f.close()


if __name__ == '__main__':
    import fire
    fire.Fire()
