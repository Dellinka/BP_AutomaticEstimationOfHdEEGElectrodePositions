"""
This script convert XML (ground truth repaired) annotations into csv files.

Used for creating csv file from annotations from CVAT - divided into folders
(creates ../Dataset/2d_annotations_csv from ../Dataset/2d_annotations_xml)
"""

import os
import xml.etree.ElementTree as ET

INPUT_DIR = os.path.join('..', 'Dataset', '2d_annotations_xml')
OUTPUT_DIR = os.path.join('..', 'Dataset', '2d_annotations_csv')

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JK", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "TN", "VV"]


def convert_annot(input_f, output_f):
    """
    This script converts XML annotations (ground truth annotation manually repaired in https://cvat.org)
    into csv file.

    :params input_f:        Path to input file
    :params output_f:       Path to output file
    """
    print(input_f, output_f)
    if not os.path.exists(os.path.dirname(output_f)):
        os.makedirs(os.path.dirname(output_f))

    f = open(output_f, 'w')
    f.write('Index; Bbox ymin xmin ymax xmax; Label; Sensor id;')

    tree = ET.parse(input_f)
    idx = 0
    for obj in tree.findall('object'):
        label = 0 if obj.find('name').text == "BLACK" else 1
        bndbox_anno = obj.find('bndbox')

        f.write('\n' + str(idx) + '; '
                + str(bndbox_anno.find('ymin').text) + ' ' + str(bndbox_anno.find('xmin').text) + ' '
                + str(bndbox_anno.find('ymax').text) + ' ' + str(bndbox_anno.find('xmax').text) + ' ' + '; '
                + str(label) + '; ')
        idx += 1
    f.close()


if __name__ == '__main__':
    for s in subjects:
        for camera_num in range(1, 12):
            input_file = os.path.join(INPUT_DIR, s, 'camera' + str(camera_num) + '.xml')
            output_file = os.path.join(OUTPUT_DIR, s, 'camera' + str(camera_num) + '.csv')
            convert_annot(input_file, output_file)
