"""


"""
import os
import sys

import numpy as np

# Import all directories
sys.path.insert(0, os.path.abspath("CameraCalibration"))
sys.path.insert(0, os.path.abspath("PointDistributionModels"))

from CameraCalibration.CameraComputation import create_average_matrices
from PointDistributionModels.evaluate import evaluate_model
from PointDistributionModels.statisticalModel import create_statistical_model
from run import run

orig_2d_dir = os.path.join('Dataset', '2d_annotations_csv_orig')
orig_3d_dir = os.path.join('Dataset', '3d_annotations_csv_orig')

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV",
            "JK", "TN"]

subjectsFiles = ["AH.gpsr", "DD.gpsr", "EN.gpsr", "HH.gpsr", "JH.gpsr", "JJ.gpsr", "JR.GPSR", "LK.gpsr",
                 "LuK.gpsr", "MB.gpsr", "MH.GPSR", "MK.gpsr", "MRB.GPSR", "RA.gpsr", "TD.gpsr", "VV.GPSR"]

err = list()
subjects_num = len(subjects) - 2
for idx in range(subjects_num):  # As last two are the subjects with different types of electrodes
    subject = subjects[0]
    subjects.pop(0)

    output_dir = os.path.join('TEST_model', subject)
    images_dir = os.path.join('Dataset', 'GPS_solved', subjectsFiles[idx], 'images')

    # print("Computation of statistical model and camera matrices on subjects ", str(subjects_all))
    create_statistical_model(subjects, orig_3d_dir)
    create_average_matrices(subjects, orig_2d_dir, orig_3d_dir)

    time_RCNN, time_TM, time_corres, time_fitting, time_model = run(images_dir=images_dir, output_dir=output_dir)
    err.append(evaluate_model(subject, output_dir, original_dir=orig_3d_dir))

    subjects.append(subject)
    print("SUBJECT {} \n RCNN {} \n TM {} \n Coorespondences {} \nFitting {} \n Model {} \n".
          format(subject, time_RCNN, time_TM, time_corres, time_fitting, time_model))
    print("{} out of {} DONE".format(idx+1, subjects_num))
    print("--------------------------------------------")

print("Leave-one-out evaluation -> All errors: ")
print(err)

print("Average {}, Minimum: {}, Maximum: {}".
      format(np.average(np.asarray(err)), np.min(np.asarray(err)), np.max(np.asarray(err))))