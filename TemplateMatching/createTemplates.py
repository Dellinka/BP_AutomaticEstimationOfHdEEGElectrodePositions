"""
Helper script for creating templates for matching. Saves all cropped images (sensors) according to
type condition ( white square, black square ... ) into ./templates/all/ directory.
Then cluster all sensors into directories and for each directory create a template as average image.

WARNING: This script creates different templates every time!
"""

import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

ANNOTATIONS_PATH = os.path.join('..', 'Dataset', '2d_annotations_xml')
IMAGES_PATH = os.path.join('..', 'Dataset', 'images_dir')
TEMPLATES_PATH = os.path.join('templates', 'all')

subjects = ["AH", "DD", "EN", "HH", "JH", "JJ", "JR", "LK",
            "LuK", "MB", "MH", "MK", "MRB", "RA", "TD", "VV"]


def propose_images(color):
    """
    Save all black and white images of sensors (only sensors images) from ANNOTATIONS_PATH annotations (GT annotations)
    into templates/all/<type> directory. Types are horizontal, vertical, square ans small square.

    :param color:       color of sensors 'BLACK' or 'WHITE'
    :return sensors, sizes
    """
    def propose_candidate(sensors, sizes, filenames, type):
        """
        Crop candidate from image and save information about it into given arrays.

        :params sensors:            Array of corresponding croped sensors
        :params sizes:              Array of corresponding sizes
        :params filenames:          Array of corresponding filenames
        :params type:               Type of sensor (black, white, square, vertical ...)

        Returns: Updated sensors, sizes and filenames
        """
        candidate = img[bbox[0]:bbox[2], bbox[1]: bbox[3]]
        if len(candidate) == 0: return sensors, sizes, filenames

        file = os.path.join(TEMPLATES_PATH, type, subject + '_' + str(idx) + '.png')
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        cv2.imwrite(file, candidate)

        sensors.append(candidate)
        sizes.append(candidate.shape)
        filenames.append(file)
        return sensors, sizes, filenames

    def average_size(sizes):
        """ Computes average size from array of sizes. """
        if len(sizes) == 0:
            return 0
        return np.mean(np.asarray(sizes), axis=0).astype(int)

    print("Proposing images")

    sensors_sq, sizes_sq, filenames_sq = [], [], []
    sensors_sq_s, sizes_sq_s, filenames_sq_s = [], [], []
    sensors_hor, sizes_hor, filenames_hor = [], [], []
    sensors_hor_s, sizes_hor_s, filenames_hor_s = [], [], []
    sensors_vert, sizes_vert, filenames_vert = [], [], []
    sensors_vert_s, sizes_vert_s, filenames_vert_s = [], [], []

    for subject in subjects:
        idx = 1
        for cam in range(1, 12):
            img = cv2.imread(os.path.join(IMAGES_PATH, subject, 'camera' + str(cam) + '.png'), cv2.IMREAD_GRAYSCALE)
            tree = ET.parse(os.path.join(ANNOTATIONS_PATH, subject, 'camera' + str(cam) + '.xml'))
            objects = tree.findall('object')
            for obj in objects:
                if obj.find('name').text == color:
                    bndbox_anno = obj.find('bndbox')
                    bbox = [int(float(bndbox_anno.find(tag).text)) for tag in ('ymin', 'xmin', 'ymax', 'xmax')]
                    h = bbox[2] - bbox[0]
                    w = bbox[3] - bbox[1]

                    # Propose squares
                    if w == h:
                        if w < 55:
                            sensors_sq_s, sizes_sq_s, filenames_sq_s = \
                                propose_candidate(sensors_sq_s, sizes_sq_s, filenames_sq_s, 'square small')
                        else:
                            sensors_sq, sizes_sq, filenames_sq = \
                                propose_candidate(sensors_sq, sizes_sq, filenames_sq, 'square')

                    # Propose horizontal rectangles
                    elif 1 / 2 * w <= h <= 3 / 4 * w:
                        if color == 'WHITE' and w <= 38:
                            sensors_hor_s, sizes_hor_s, filenames_hor_s = \
                                propose_candidate(sensors_hor_s, sizes_hor_s, filenames_hor_s, 'horizontal_small')
                        else:
                            sensors_hor, sizes_hor, filenames_hor =\
                                propose_candidate(sensors_hor, sizes_hor, filenames_hor, 'horizontal')

                    # Propose vertical rectangles
                    elif 1 / 2 * h <= w <= 3 / 4 * h:
                        if color == 'WHITE' and h <= 38:
                            sensors_vert_s, sizes_vert_s, filenames_vert_s = \
                                propose_candidate(sensors_vert_s, sizes_vert_s, filenames_vert_s, 'vertical_small')
                        else:
                            sensors_vert, sizes_vert, filenames_vert = \
                                propose_candidate(sensors_vert, sizes_vert, filenames_vert, 'vertical')

                    idx += 1

    ret = {
        'square': {'sensors': sensors_sq, 'shape': average_size(sizes_sq), 'filenames': filenames_sq},
        'square_small': {'sensors': sensors_sq_s, 'shape': average_size(sizes_sq_s), 'filenames': filenames_sq_s},
        'horizontal': {'sensors': sensors_hor, 'shape': average_size(sizes_hor), 'filenames': filenames_hor},
        'horizontal_small': {'sensors': sensors_hor_s, 'shape': average_size(sizes_hor_s), 'filenames': filenames_hor_s},
        'vertical': {'sensors': sensors_vert, 'shape': average_size(sizes_vert), 'filenames': filenames_vert},
        'vertical_small': {'sensors': sensors_vert_s, 'shape': average_size(sizes_vert_s), 'filenames': filenames_vert_s}
    }

    return ret


def histogram_equalization(img):
    """
    Compute image with equalized histogram for better  templates and better results in
    template matching algorithm.

    :param img:     Image as numpy array

    Returns:    Image with equalized histogram
    """
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[img]


def preprocess_images(sensors, shape):
    """
    Preprocess images of sensors for easier clustering. Preprocessing consists of
    image resizing and histogram equalization.

    :params sensors:        All images of sensors in dictionary
    :params shape:          Desired shape of images

    Returns:    Preprocessed images on sensors

    """
    preprocessed = []

    # Resize all images and save in preprocessed list
    for s in sensors:
        # Resize images to average shape
        img = np.asarray(cv2.resize(s, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC))
        img = histogram_equalization(img)
        img = img / 255
        preprocessed.append(np.reshape(img, -1))

    return np.asarray(preprocessed)


def kmean_clustering(sensors, filenames, type, n_clusters=9):
    """
    Cluster images from the same category/type into specified number of clusters
    using scikit learn library.

    :params sensors:            Images of sensors as numpy array(image_num, pixels_in_array)
    :params filenames:          Filenames of all sensors
    :params type:               Type of sensors to be clustered
    :params n_clusters:         Number of clusters

    Returns: Filenames of each cluster in numpy array
    """
    print("Clustering started")
    cluster_dirs = []
    kmeans = KMeans(min(n_clusters, len(sensors)))
    kmeans.fit(sensors)

    for idx, label in enumerate(kmeans.labels_):
        output = os.path.join(TEMPLATES_PATH, type + '_cluster' + str(label))
        if not os.path.exists(output):
            os.makedirs(output)

        os.replace(filenames[idx], os.path.join(output, os.path.basename(filenames[idx])))
        cluster_dirs.append(type + '_cluster' + str(label))

    # Remove empty directories
    os.rmdir(os.path.dirname(filenames[0]))

    return np.unique(cluster_dirs)


def average_image(type):
    """
    Computation and save of average image with equalized histogram from given cluster.

    :param type:         Type of cluster & Name of directory with images from this cluster
    """
    print("Creating average image of cluster ", type)
    dir = os.path.join(TEMPLATES_PATH, type)
    images_file = [f for f in listdir(dir) if isfile(join(dir, f))]

    sizes = []
    images = list()

    for file in images_file:
        img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        sizes.append(img.shape)

    average_shape = np.mean(np.asarray(sizes), axis=0).astype(int)
    h, w = average_shape[0], average_shape[1]

    for file in images_file:
        img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = histogram_equalization(img)
        images.append(img)

    return np.mean(images, axis=0).astype(np.uint8)


if __name__ == '__main__':
    print("Purpose of this file is creation of templates for template matching algorithm. \n"
          "However the recomputation templates creates different templates every time!\n"
          "If you still want to continue, you have to uncomment the "
          "computation at the end of the createTemplates.py file")

    """
    colors = ['WHITE', 'BLACK']
    for c in colors:
        all_sensors = propose_images(c)

        for type in all_sensors:
            sensors_info = all_sensors[type]
            if len(sensors_info['sensors']) == 0:
                print("No sensors in ", type)
                continue

            # Number of clusters is computed from number of images such as each in each cluster is average of 200 images
            n_clusers = max(min(4, len(sensors_info['sensors'])), len(sensors_info['sensors']) // 200)
            print(type, ' ', n_clusers)

            sensors_imgs = preprocess_images(sensors_info['sensors'], sensors_info['shape'])
            cluster_dirs = kmean_clustering(sensors_imgs,
                                            sensors_info['filenames'],
                                            c + '_' + type,
                                            n_clusters=n_clusers)
            for cl_dir in cluster_dirs:
                avg_img = average_image(cl_dir)
                cv2.imwrite(os.path.join('templates', type + '.png'), avg_img)
    """