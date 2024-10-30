"""
Script for creating XML dataset in VOC format from detectedSensors.xml from provided GPS_solved directory

This code was used for creating annotations used in https://cvat.org for manual detection.
"""
import os

from VOC_writer import Writer
import xml.etree.ElementTree as ET
import cv2


camera_nums = 11
data_dir = os.path.join('..', 'Dataset', 'GPS_solved')
output_path = "~/detectedVOC"
subjectsFiles = ["AH.gpsr", "DD.gpsr", "EN.gpsr", "HH.gpsr", "JH.gpsr", "JJ.gpsr", "LK.gpsr", "LuK.gpsr", "MB.gpsr",
                 "MK.gpsr", "RA.gpsr", "TD.gpsr", "TN.gpsr", "JK.GPSR", "JR.GPSR", "MH.GPSR", "MRB.GPSR", "VV.GPSR"]
blackSensors = [-3, 31, 36, 67, 72, 111, 114, 119, 168, 173, 199, 219, 224, 234, 237, 244, 247, 257]


def tiff2png(tiff_file, png_file):
    """
    Save image from tiff format into png

    :param tiff_file            path to image in tiff format (ex. imgs/i.tiff)
    :param png_file             path where to save png image (ex. imgs_png/i.png)
    """
    tiff_image = cv2.imread(tiff_file)
    ret = cv2.imwrite(png_file, tiff_image)

    if not os.path.exists(os.path.dirname(png_file)):
        os.makedirs(os.path.dirname(png_file))

    if not ret:
        print(png_file, " ERROR SAVING")


def createVOCxml(subject_dir, camera_num, output_dir, filename):
    """
    From '/detectedSensors.xml' create PASCAL VOC format xml and save

    :param subject_dir          path to solved subject directory
    :param camera_num           number of camera
    :param output_dir           path to directory where to save PASCAL VOC format xml
    :param filename             name of the final file  (without file extension)
    """
    im = cv2.imread(os.path.join(subject_dir, 'images', 'camera' + str(camera_num) + '.tiff'))
    img_width = im.shape[1]
    img_height = im.shape[0]

    VOC_writer = Writer(img_width, img_height, filename=filename + ".png")

    tree = ET.parse(os.path.join(subject_dir, 'detectedSensors.xml'))
    root = tree.getroot()
    camera = root[camera_num - 1]
    if camera.attrib['number'] != str(camera_num):
        raise Exception('MY EXCEPTION: Reading wrong camera from detectedSensors.xml')

    # Create xml element for each sensor in camera
    for sensorR in camera:
        if float(sensorR.find('width').text) == 0 or float(sensorR.find('height').text) == 0:
            continue
        if float(sensorR.attrib['id']) == -3:
            continue

        x = float(sensorR.find('x').text)
        y = float(sensorR.find('y').text)
        width = float(sensorR.find('width').text)
        height = float(sensorR.find('height').text)

        xmin = x - width/2
        ymin = y - height/2
        xmax = x + width/2
        ymax = y + height/2
        label = 'BLACK' if int(sensorR.attrib['id']) in blackSensors else 'WHITE'

        VOC_writer.addObject(label, xmin, ymin, xmax, ymax)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    VOC_writer.save(os.path.join(output_dir, filename + ".xml"))


if __name__ == '__main__':
    if output_path[0] == "~":
        print("You have to specify the output directory.")
        exit()

    for subjectFile in subjectsFiles:
        for cam in range(1, camera_nums+1):
            tiff2png(os.path.join(data_dir, subjectFile, "images", "camera" + str(cam) + '.tiff'),
                     os.path.join(output_path, 'images', subjectFile.split(".")[0] + '_camera' + str(cam) + '.png'))

            createVOCxml(os.path.join(data_dir, subjectFile),
                         cam, output_path,
                         subjectFile.split(".")[0] + "_camera" + str(cam))

        print(subjectFile.split(".")[0], " annotations saved into " + output_path,
              subjectFile.split(".")[0] + "_camera**.xml")
