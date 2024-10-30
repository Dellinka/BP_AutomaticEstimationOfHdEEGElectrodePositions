"""
This script computes camera constraints for finding correspondences between sensors predictions. Constraints are
computed from training dataset from detectedSensors.xml.
"""
import os
import xml.etree.ElementTree as ET

DATA_DIR = os.path.join('..', 'Dataset', 'GPS_solved')
OUTPUT_FILE = os.path.join('..', 'Config', 'cameraConstraints.txt')


def compute_constraints(filename):
    """
    The camera constraints from detectedSensors.xml file are computed in this function.

    :param filename:        Path to file with detectedSensors.xml

    Returns:                Constraints as dictionary with 11 sets {'1' : {2, 3, 4, 5, 6}, '2' : ...}
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # Read sensors in all cameras
    visible_sensors = {}
    for camera in root:
        cam_num = camera.attrib['number']
        visible_sensors[cam_num] = set()
        for sensorR in camera:
            visible_sensors[cam_num].add(int(sensorR.attrib['id']))

    # Compute constraints
    constraints = {}
    for i in visible_sensors:
        constraints[i] = set()
        for j in visible_sensors:
            if j == i: continue
            if len(visible_sensors[i].intersection(visible_sensors[j])) > 0:
                constraints[i].add(int(j))

    return constraints


def save_constraint(output_file, constraints):
    """
    Saves computed constraints into output file.

    :param output_file:             Path to output file
    :param constraints:             Dictionary of 11 constraints for each camera (from compute_constraints() function)
    """
    f = open(output_file, 'w')

    for cam in constraints:
        visible_cameras = ''
        for visible in sorted(constraints[cam]):
            visible_cameras += str(visible) + " "

        f.write(str(cam) + ": " + visible_cameras + "\n")

    f.close()


if __name__ == '__main__':
    final = None
    subjectsFiles = ["AH.gpsr", "DD.gpsr", "EN.gpsr", "HH.gpsr", "JH.gpsr", "JJ.gpsr", "LK.gpsr", "LuK.gpsr", "MB.gpsr",
                     "MK.gpsr", "RA.gpsr", "TD.gpsr", "TN.gpsr", "JK.GPSR", "JR.GPSR", "MH.GPSR", "MRB.GPSR", "VV.GPSR"]

    for subject in subjectsFiles:
        c = compute_constraints(os.path.join(DATA_DIR, subject, 'detectedSensors.xml'))
        if final is None:
            final = c
        else:
            for cam in final:
                final[cam] = c[cam].union(final[cam])

    save_constraint(OUTPUT_FILE, final)
    print("Constraints saved into " + OUTPUT_FILE)
