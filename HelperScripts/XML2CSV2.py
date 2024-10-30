"""
This script also convert XML (ground truth) annotations into csv files as XML2CSV.py but as
ground truth annotations uses detectedSensors.xml (2D) from GPS_solved directory .

Used for CameraCalibration part of the project.
(Creates ../Dataset/2d_annotations_csv_orig from ~/GPS_solved/<subject>/detectedSensors.xml)
"""

import os
import xml.etree.ElementTree as ET


camera_nums = 11
data_dir = os.path.join('..', 'Dataset', 'GPS_solved')           # Path to provided GPS_solved directory
output_path = os.path.join('..', 'Dataset', '2d_annotations_csv_orig')
subjectsFiles = ["AH.gpsr", "DD.gpsr", "EN.gpsr", "HH.gpsr", "JH.gpsr", "JJ.gpsr", "LK.gpsr", "LuK.gpsr", "MB.gpsr",
                 "MK.gpsr", "RA.gpsr", "TD.gpsr", "TN.gpsr", "JK.GPSR", "JR.GPSR", "MH.GPSR", "MRB.GPSR", "VV.GPSR"]


def createCSV(subject_dir, camera_num, output_file):
    """
        From '~detectedSensors.xml' create CSV file of all sensors for provided camera number

        :param subject_dir          path to solved subject directory
        :param camera_num           number of camera
        :param output_file          path to output file
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    f = open(output_file, 'w')
    f.write('Sensor id; Sensor center x; Sensor center y;')

    tree = ET.parse(os.path.join(subject_dir, 'detectedSensors.xml'))
    root = tree.getroot()
    camera = root[camera_num - 1]
    if camera.attrib['number'] != str(camera_num):
        raise Exception('MY EXCEPTION: Reading wrong camera from detectedSensors.xml')

    for sensorR in camera:
        id = int(sensorR.attrib['id'])
        if id < 0 or id > 257: continue

        x = float(sensorR.find('x').text)
        y = float(sensorR.find('y').text)

        f.write('\n'
                + str(id) + ';'
                + str(x) + ';'
                + str(y) + ';')

    f.close()


if __name__ == '__main__':
    for subjectFile in subjectsFiles:
        for cam in range(1, camera_nums+1):
            createCSV(os.path.join(data_dir, subjectFile), cam,
                      os.path.join(output_path, subjectFile.split(".")[0], "camera" + str(cam) + ".csv"))

        print(subjectFile.split(".")[0], " annotations saved into " +
              os.path.join(output_path, subjectFile.split(".")[0], "camera**.csv")
              )
