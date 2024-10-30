"""
Final script for localization of eeg electrodes using a system of cameras.

This script is made of different parts described more in README.md file. All predictions from all
parts are stored in specified output file. Predictions from RCNN and template matching and computed correspondences 
can be visualized using scripts from VisualizationScripts directory.

FasterRCNN part is called using subprocess due to the parts of RCNN algorithms being in different directories -
standard imports could be used using 'sys.path.insert()' in all files in all subdirectories.


:param images_dir           Path to directory with 11 images in tiff format used for prediction
                            Images in directory must be named as camera<i>.tiff for <i> in range 1 to 11
                            (--images_dir='../GPS_solved/AH.gpsr/images/')
:param output_dir           Path to output directory
                            (--output_dir='result/')
"""

import os
import sys
import cv2
import datetime
import subprocess

# Import all directories
sys.path.insert(0, os.path.abspath("TemplateMatching"))
sys.path.insert(0, os.path.abspath("PointMatching"))
sys.path.insert(0, os.path.abspath("PointDistributionModels"))

import TemplateMatching.main as TemplateMatching
import PointMatching.main as PointMatching
import PointDistributionModels.main as PointDistributionModels

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def tiff2png(tiff_file, png_file):
    """
    Save image from tiff format into png

    :param tiff_file            path to image in tiff format (ex. imgs/i.tiff)
    :param png_file             path to save png image (ex. imgs_png/i.png)
    """
    tiff_image = cv2.imread(tiff_file)
    ret = cv2.imwrite(png_file, tiff_image)
    if not ret:
        print("ERROR: Saving image {} as png".format(tiff_file), file=sys.stderr)


def create_directory(path):
    """
    Creates given directory or path to file if it does not exist.

    :param path:                Path or directory to be created
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def run(**kwargs):
    """
    :param kwargs:      (** images_dir and ** output_dir) more information in the beginning of this script
    """
    # Check input arguments
    if kwargs['images_dir'] is None:
        print("ERROR: images_dir not specified", file=sys.stderr)
        exit()
    if kwargs['output_dir'] is None:
        print("ERROR: output_dir not specified", file=sys.stderr)
        exit()

    # Check directory with images
    if not os.path.exists(kwargs['images_dir']):
        print("ERROR: Given path to images does not exist", file=sys.stderr)
        exit()

    # Create output directory if it does not exist
    create_directory(kwargs['output_dir'])

    # --------------- Converting tiff images into png format ----------------
    for idx in range(1, 12):
        create_directory(os.path.join(kwargs['output_dir'], 'images'))
        tiff2png(os.path.join(kwargs['images_dir'], 'camera' + str(idx) + '.tiff'),
                 os.path.join(kwargs['output_dir'], 'images', 'camera' + str(idx) + '.png'))
    print("DONE: All tiff images converted into png format")

    # --------------- Create prediction of sensors via RCNN ----------------
    print("Creating predictions of sensors via RCNN ...")

    start = datetime.datetime.now()
    create_directory(os.path.join('RCNN'))
    cmd = ['python', 'FasterRCNN/main.py', 'predict_and_save',
           '--input_dir=' + os.path.join(kwargs['output_dir'], 'images'),
           '--output_dir=' + os.path.join(kwargs['output_dir'], 'RCNN')]
    p = subprocess.Popen(cmd)
    p.wait()
    end = datetime.datetime.now()

    time_RCNN = (end - start).total_seconds() * 1000
    print("DONE: All predictions created in {} ms".format(time_RCNN))

    # --------------- Matching templates on predicted sensors --------------
    print("Matching templates on predicted sensors ...")

    start = datetime.datetime.now()
    create_directory(os.path.join(kwargs['output_dir'], 'TemplateMatching'))
    TemplateMatching.fit_templates_and_save(
        images_dir=os.path.join(kwargs['output_dir'], 'images'),
        predictions_dir=os.path.join(kwargs['output_dir'], 'RCNN'),
        output_dir=os.path.join(kwargs['output_dir'], 'TemplateMatching'))
    end = datetime.datetime.now()

    time_TM = (end - start).total_seconds() * 1000
    print("DONE: All templates matched - coordinates predicted in {} ms".format(time_TM))

    # --------------- Find correspondences between predictions --------------
    print("Finding correspondences ...")

    start = datetime.datetime.now()
    create_directory(os.path.join(kwargs['output_dir'], 'Correspondences'))
    PointMatching.create_and_save_correspondences(
        predictions_dir=os.path.join(kwargs['output_dir'], 'TemplateMatching'),
        camera_matrices=os.path.join('Config', 'CameraMatrices'),
        camera_constraint=os.path.join('Config', 'cameraConstraints.txt'),
        output_dir=os.path.join(kwargs['output_dir'], 'Correspondences'))
    end = datetime.datetime.now()

    time_corres = (end - start).total_seconds() * 1000
    print("DONE: Correspondences created and saved in {} ms".format(time_corres))

    # --------------- Fitting statistical model on predicted data --------------
    print("Fitting statistical model ...")

    start = datetime.datetime.now()
    create_directory(os.path.join(kwargs['output_dir'], 'FittedModel'))
    PointDistributionModels.fit_and_save_model(
        predicted_file=os.path.join(kwargs['output_dir'], 'Correspondences', 'corresponding_points.csv'),
        model_dir=os.path.join('Config', 'StatisticalModel'),
        output_dir=os.path.join(kwargs['output_dir'], 'FittedModel'))
    end = datetime.datetime.now()

    time_fitting = (end - start).total_seconds() * 1000
    print("DONE: Model fitted in {} ms".format(time_fitting))

    # --------------- Computing 3D model from fitted and predicted model -----------
    print("Creating 3D model...")

    start = datetime.datetime.now()
    PointDistributionModels.create_and_save_3d_model(
        predicted_file=os.path.join(kwargs['output_dir'], 'Correspondences', 'corresponding_points.csv'),
        model_dir=os.path.join(kwargs['output_dir'], 'FittedModel'),
        output_dir=os.path.join(kwargs['output_dir']))
    end = datetime.datetime.now()

    time_model = (end - start).total_seconds() * 1000
    print("DONE: Final 3D model created in {} ms".format(time_model))

    return time_RCNN, time_TM, time_corres, time_fitting, time_model


if __name__ == '__main__':
    import fire
    fire.Fire()
