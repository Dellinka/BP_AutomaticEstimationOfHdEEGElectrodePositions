import numpy as np


class CorrespondingPoints:
    """
    Wrapper for consistent / possibly corresponding points. These points have the same color
    and their projections from 3D point are consistent (differ in less than 10 px).

    Note:  self.points is a set of possibly corresponding points as tuples (documentation => set 'C')
           self.points =  {camera: (x, y),                    Corresponding points through different images
                           ... })
    """

    def __init__(self, sensor_color):
        self.color = sensor_color
        self.coord_3d = np.ones(3)
        self.points = {}
        self.points_reprojected = {}
