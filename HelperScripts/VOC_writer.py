"""
Implementation of PASCAL VOC Writer according to https://github.com/AndrewCarterUK/pascal-voc-writer
Changed mainly because in our dataset can be some unknown parameters.
"""

import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, width, height, filename=None, depth='', database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('pascal_voc_writer', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        filename = 'Unknown' if filename is None else filename

        self.template_parameters = {
            'filename': filename,
            'folder': os.path.basename(os.path.dirname(filename)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)
