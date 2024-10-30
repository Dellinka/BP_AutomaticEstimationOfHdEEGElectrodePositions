import os

import numpy as np
import xml.etree.ElementTree as ET

from . import util
from .util import read_image


class EEGDataset:
    def __init__(self, data_dir, type):
        self.label_names = EEG_BBOX_LABEL_NAMES
        self.data_dir = data_dir
        # -1000 manual annotation, unknown id for black ! 1000 manual annotation, unknown id for white
        self.blackSensors = [-3, 31, 36, 67, 72, 111, 114, 119, 168, 173, 199, 219, 224, 234, 237, 244, 247, 257, -1000]
        self.ids_original = [id_.strip() for id_ in open(data_dir + EEG_DATA_TYPE[type])]
        self.ids_negative = [id_.strip() for id_ in open(data_dir + EEG_DATA_TYPE['negative'])]
        self.type = type

    def __len__(self):
        """Returns length of dataset

        Finale train dataset consist of original, horizontally flipped and negative images.
        Length of dataset is length of original images multiplied by 2 (original + flipped)
        and length of negative dataset -> special implementation of data augmentation.
        For Test and Eval dataset returns number of original images only.

        Returns:
            integer - number of images in dataset

        """
        if self.type == 'train':
            return 2*len(self.ids_original) + len(self.ids_negative)         # Flipped + Flipped2 (changed Test dataset)
            # return 4 * len(self.ids_original) + 4 * len(self.ids_negative) # Flipped + Blur + Negative
            # return 4 * len(self.ids_original)                              # FlippedBlur
            # return len(self.ids_original) + len(self.ids_negative)         # Negative images
            # return len(self.ids_original)                                  # Repaired + Bad

        else:
            return len(self.ids_original)

    def get_example(self, idx):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            idx (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        # Blur + Negative + Flip
        # file = self.ids_original[idx % len(self.ids_original)] if idx < 4 * len(self.ids_original) \
        #    else self.ids_negative[(idx - 4 * len(self.ids_original)) % len(self.ids_negative)]

        # Negative images
        file = self.ids_original[idx % len(self.ids_original)] if idx < 2 * len(self.ids_original) \
                else self.ids_negative[(idx - 2 * len(self.ids_original)) % len(self.ids_negative)]

        # file = self.ids_original[idx % len(self.ids_original)]
        try:
            anno = ET.parse(
                os.path.join(self.data_dir, '2d_annotations_xml', file + '.xml'))
            bbox = list()
            label = list()
            difficult = list()

            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')

                bbox.append([
                    float(bndbox_anno.find(tag).text)
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

                name = obj.find('name').text.upper()
                label.append(EEG_BBOX_LABEL_NAMES.index(name))
                difficult.append(0)
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

            # Load image
            img_file = os.path.join(self.data_dir, 'images_dir', file + '.png')
            img = read_image(img_file, color=True)
            original_num = len(self.ids_original)

            # Return negative image
            if idx >= 2 * original_num:
                img = 255 - img
                label = abs(1 - label)

                """
                # Return flipped image
                if 2 * original_num + len(self.ids_negative) <= idx:
                    img = img[:, :, ::-1]
                    _, H, W = img.shape
                    bbox = util.flip_bbox(bbox, (H, W), x_flip=True)
                """

            else:
                # Return flipped image
                if original_num <= idx:
                    img = img[:, :, ::-1]
                    _, H, W = img.shape
                    bbox = util.flip_bbox(bbox, (H, W), x_flip=True)

            """  Part for FLipped + Blur + Negative images
            # Return negative image
            if idx >= 4 * original_num:
                img = 255 - img
                label = abs(1 - label)

                # Return flipped image + blur
                if 4 * original_num + len(self.ids_negative) <= idx < 4 * original_num + 3 * len(self.ids_negative):
                    img = img[:, :, ::-1]
                    _, H, W = img.shape
                    bbox = util.flip_bbox(bbox, (H, W), x_flip=True)
                if 4 * original_num + 2 * len(self.ids_negative) <= idx:
                    img = np.uint8(img.transpose((1, 2, 0)))
                    img = Image.fromarray(img)
                    img = np.asarray(img.filter(ImageFilter.GaussianBlur(1.25)))
                    img = img.transpose((2, 0, 1))

            else:
                # Return flipped image + blur (This part alone is ok for 'all' except Flipped+Blur+Negative)
                if original_num <= idx < 3 * original_num:
                    img = img[:, :, ::-1]
                    _, H, W = img.shape
                    bbox = util.flip_bbox(bbox, (H, W), x_flip=True)
                if 2 * original_num <= idx:
                    img = np.uint8(img.transpose((1, 2, 0)))
                    img = Image.fromarray(img)
                    img = np.asarray(img.filter(ImageFilter.GaussianBlur(1.25)))
                    img = img.transpose((2, 0, 1))
            """

            return img, bbox, label, difficult, file
        except Exception as e:
            print(str(e), " in ", file)

    __getitem__ = get_example


EEG_BBOX_LABEL_NAMES = (
    'BLACK',
    'WHITE'
)

EEG_DATA_TYPE = {
    'train': os.path.join('RCNN', 'datasetListTrain.txt'),
    'test': os.path.join('RCNN', 'datasetListTest.txt'),
    'negative': os.path.join('RCNN', 'datasetListNegative.txt')
}
