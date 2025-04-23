# All world objects are defined here:
#   Lane (road), Line (check points), Traffic signs, Apriltags, ...

import numpy as np
import apriltag


class Observation:
    def __init__(self):
        self.label = 'world_object'
        self.box = np.array([0, 0, 0, 0])
        self.center = np.array([0, 0])
        self.center_ud = np.array([0, 0])
        self.distance = 0.0
        self.Xb = np.array([0., 0., 0.])

    def get_center(self):
        return self.center

    def set_object(self, obj):
        print('Base set_obj method')


class Line(Observation):
    def __init__(self):
        super().__init__()
        self.color = 'NA'
        self.line = None

    def set_object(self, obj, color='NA'):
        self.line = obj
        self.color = color
        # todo: compute center


# Define a class to receive the characteristics of each line detection
class LaneLine(Line):
    def __init__(self):
        # was the line detected in the last iteration?
        super().__init__()
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_info = None
        self.curvature = None
        self.deviation = None


class Apriltag(Observation):
    def __init__(self, tag_obj: apriltag.Detection = None):
        super().__init__()
        self.apriltag = tag_obj
        self.center = tag_obj.center

    def set_object(self, obj: apriltag.Detection):
        self.apriltag = obj
        self.center = obj.center
        # todo: set other attributes like bbox and ...


class TrafficSign(Observation):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    print('World Object\'s Main')
