import os
from os import listdir
from os.path import isfile, join

from data.dataset import TestDataset
from main import predict_and_save

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from torch.utils import data as data_
from matplotlib.ticker import MaxNLocator

import numpy as np
from train import eval_net
import matplotlib.pyplot as plt
import scipy.optimize

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


def compute_thresholds(net, sc_num=45, nms_num=15):
    """
    EDIT: Found out that MAP value is not suitable for this kind of problem -> at the end was not used
    This function computes and plot MAP for different values of confidence score and nms thresholds.
    USed for optimization of these parameters such as MAP value is maximized. (Later found out that
    MAP value is not as valuable information as we thought - often is maximal for confidence score 0
    for which all predictions are taken as final = very low precision! but on the other hand higher recall)

    Args:
        net:            Trained neural network
        sc_num:         Number of confidence score to try
        nms_num:        Number of nms thresholds to try

    Returns:

    """
    def plot3D_thresholds(data):
        data = np.array(data)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2])
        cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        fig.colorbar(surf, cax=cbaxes)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Confidence score treshold')
        ax.set_ylabel('NMS Treshold')
        ax.set_zlabel('Mean average precision')

        fig.tight_layout()

        plt.savefig('thresholds_flippedRandom_close.png', dpi=300)

    score_lst = np.linspace(0.05, 0.9, sc_num)
    nms_lst = np.linspace(0.05, 0.35, nms_num)
    data = []
    i = 0
    for score in score_lst:
        for nms in nms_lst:
            map = eval_net(data, net, use_preset=False, nms=nms, score=score)['map']
            data.append([score, nms, map])
            i += 1
        print('{} out of {} computed'.format(i, sc_num*nms_num))

    # Save computed data
    f = open('data.csv', 'w')
    f.write('Score; NMS; MAP')
    for i in range(len(data)):
        f.write(str(data[i][0]) + '; ' + str(data[i][1]) + '; ' + str(data[i][2]) + ';\n')
    f.close()

    plot3D_thresholds(data)
    return data


def predict_all():
    """
    This is helper function used for predicting bounding boxes for all subjects from dataset.
    """
    path_imgs = os.path.join('..', 'Dataset', 'images_dir')
    output = os.path.join('..', 'GeneratedData', 'RCNN')

    for s in subjects:
        predict_and_save(input_dir=os.path.join(path_imgs, s), output_dir=os.path.join(output, s))
        print(s, " DONE")


def get_optimal_params(network, data_loader):
    params = {}

    def optimal_nms(score_fixed, net, dataset):
        """
        Find optimal nms (nms_opt_tmp) threshold for fixed confidence score such as
        the precision  in both classes is higher than 0.8
        Return 1 - recall (computed with fixed score and optimal nms)
        """

        def evaluate(nms, score):
            """
            Evaluate network result on fixed nms and score threshold.
            Return 1 - recall = value to be minimized - for recall to be maximized
            """
            ret = eval_net(dataset, net, use_preset=False, nms=nms, score=score)
            prec = min(ret['precision']['0'], ret['precision']['1'])
            rec = (ret['recall']['0'] + ret['recall']['1']) / 2
            rec = -rec/prec if prec < 0.8 else rec
            print('Score {}, nms {}, recall {}, precision B, W {}, {}'.
                  format(score, nms, rec, ret['precision']['0'], ret['precision']['1']))
            return 1 - rec

        # Optimal nms for fixed score
        nms_opt_tmp = scipy.optimize.fminbound(lambda nms: evaluate(nms, score_fixed), 0, 0.4, xtol=0.01)

        data = eval_net(dataset, net, use_preset=False, nms=nms_opt_tmp, score=score_fixed)
        precision = min(data['precision']['0'], data['precision']['1'])
        recall = (data['recall']['0'] + data['recall']['1']) / 2
        recall = -recall/precision if precision < 0.8 else recall

        print('------------------------')
        print('Score {}, OPTIMAL nms {}, recall {}, precision B, W {}, {}'.
              format(score_fixed, nms_opt_tmp, recall, data['precision']['0'], data['precision']['1']))
        print('------------------------')
        params[(score_fixed, nms_opt_tmp)] = recall
        return 1 - recall

    scipy.optimize.fminbound(lambda score: optimal_nms(score, network, data_loader), 0, 1, xtol=0.01)
    (score_opt, nms_opt) = max(params, key=params.get)
    nms_opt = round(nms_opt, 5)
    score_opt = round(score_opt, 5)

    return {'nms': nms_opt, 'score': score_opt}


if __name__ == '__main__':
    # Load pretrained network
    faster_rcnn = FasterRCNNVGG16(n_fg_class=2, anchor_scales=[4, 8, 16, 32], ratios=[0.8, 1, 1.25])
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(os.path.join('checkpoints', 'flipped2.zip'))
    print('load network')
    trainer.eval()

    # Load dataset
    dataset = TestDataset(opt)
    dataloader_test = data_.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    #opt_params = get_optimal_params(faster_rcnn, dataloader_test)
    #print(opt_params)

    #ret = eval_net(dataloader_test, faster_rcnn, use_preset=False, nms=0.12192, score=0.20665, visualize=False)
    #print(ret)

    # predict_all()

