"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import math
import numpy as np
import cv2
from argparse import Namespace
from car_utils import get_car_path, get_radius, get_car_line_mark, WHEEL_STEER_RATIO

FILTER_NEGATIVE_POINTS = True
FILTER_NONMONOTONIC_POINTS = True

CAR_L = 2.634  # Wheel base
CAR_T = 1.733  # Tread
MIN_TURNING_RADIUS = 5.
MAX_STEER = 500

CAMERA_POSITION = [0, 1.657, 1.542276316]
CAMERA_MATRIX = [
    np.array([[1173.122620, 0.000000, 969.335924],
              [0.000000, 1179.612539, 549.524382],
              [0., 0., 1.]])
]
RVEC = np.array([0.04, 0.0, 0.0])
TVEC = np.array([0.0, 0.0, 0.0])

DISTORTION = np.array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])

DEFAULT_CFG = {
    'center_color': (0, 255, 0),
    'center_width': 4,
    'side_color': (0, 255, 255),
    'side_width': 2,
    'curve_length': 30.0,
    'initial_steer': -1994.999999999999971,  # for this steer we have a straight line of trajectory

    'rvec': RVEC,
    'tvec': TVEC,
    'camera_matrix': CAMERA_MATRIX[0],
    'camera_position': CAMERA_POSITION,
    'distortion': DISTORTION,
    'mark_count': 10,
    'start_dist': 6.0,
    'gap_dist': 1.0,
    'distance_mark': [3., 5., 10., 20., 30.]
}


class TurnRadius:
    def __init__(self, cfg):
        self.car_l = cfg.car_l
        self.car_t = cfg.car_t
        self.min_turning_radius = cfg.min_turning_radius

        self.num_points = num_points = 400
        max_wheel_angle = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
        self.angles = np.linspace(-max_wheel_angle, max_wheel_angle, num_points)

    def get_car_path(self, steer_factor, distance=20):
        """
        :param steer_factor: [-1, 1] (max left max right)
        :return:
        """
        num_points = self.num_points
        idx = np.clip(int(num_points/2. * steer_factor + num_points/2.), 0, num_points-1)
        r = get_radius(self.angles[idx])
        c, lw, rw = get_car_path(r, distance=distance)
        return c, lw, rw


class TrajectoryVisualizer:
    """
        Class that takes input a configuration file that contain information about line color and
        width, parameters about the camera and other data about the distance marks.
        Provides a method that projects the trajectory of the car given a steering angle on an
        image based on parameters from configuration.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rvec = np.array(cfg.rvec)
        self.center_color = cfg.center_color
        self.center_width = cfg.center_width
        self.side_color = cfg.side_color
        self.side_width = cfg.side_width
        self.curve_length = cfg.curve_length
        self.initial_steer = cfg.initial_steer
        self.tvec = np.array(cfg.tvec)
        self.camera_matrix = np.array(cfg.camera_matrix)
        self.camera_position = np.array(cfg.camera_position)
        self.distortion = np.array(cfg.distortion)
        self.mark_count = cfg.mark_count
        self.start_dist = cfg.start_dist
        self.gap_dist = cfg.gap_dist
        self.distance_mark = cfg.distance_mark

    def project_points_on_image_space(self, points_3d):
        rvec = self.rvec
        tvec = self.tvec
        camera_matrix = self.camera_matrix
        distortion = self.distortion

        points = np.array(points_3d).astype(np.float32)
        points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
        points = points.astype(np.int)
        points = points.reshape(-1, 2)
        return points

    def render_line(self, image, points, color=None, thickness=None,
                    filter_negative=FILTER_NEGATIVE_POINTS,
                    filter_nonmonotonic=FILTER_NONMONOTONIC_POINTS):
        points = self.project_points_on_image_space(points)

        h, w, _ = image.shape

        if filter_negative:
            idx = 0
            while (points[idx] < 0).any() or points[idx+1][1] < points[idx][1]:
                idx += 1
                if idx >= len(points) - 1:
                    break

            points = points[idx:]
        else:
            points = points[(points >= 0).all(axis=1)]

        # TODO Check validity - expect monotonic modification on x
        # monotonic decrease on y
        if len(points) <= 0:
            return image

        prev_x, prev_y = points[0]
        idx = 1

        if len(points) > 1 and filter_nonmonotonic:
            while points[idx][1] < prev_y:
                prev_x, prev_y = points[idx]
                idx += 1
                if idx >= len(points):
                    break

            valid_points = []
            while points[idx][0] >= 0 and points[idx][0] < w and idx < len(points)-1:
                valid_points.append([points[idx][0], points[idx][1]])
                idx += 1
        else:
            valid_points = [[prev_x, prev_y]]

        if filter_nonmonotonic:
            points = np.array(valid_points)

        points[:, 0] = np.clip(points[:, 0], 0, w)
        points[:, 1] = np.clip(points[:, 1], 0, h)
        for p in zip(points, points[1:]):
            image = cv2.line(image, tuple(p[0]), tuple(p[1]), color=color, thickness=thickness)

        return image

    def detect_indexes_on_lane_points_for_distance_marks(self, mark_count, start_dist, dist_gap):
        initial_steer = self.initial_steer
        curve_length = self.curve_length

        c, lw, rw = get_car_path(initial_steer, distance=curve_length)
        lw = self.add_3rd_dim(lw)

        def create_horizontal_line_at_depth(distance_from_camera, left_limit=-CAR_T/2, right_limit=CAR_T/2, n=2):
            x = np.expand_dims(np.linspace(left_limit, right_limit, num=n), axis=1)
            y = np.ones((n, 1))  # * CAMERA_HEIGHT - TODO some hardcoded value
            z = np.ones((n, 1)) * distance_from_camera
            xy = np.concatenate((x, y), axis=1)
            xyz = np.concatenate((xy, z), axis=1)
            return xyz

        def get_idx_closest_point_to(points, point):
            dists = list(map(lambda x : np.linalg.norm(x - point), points))
            min_idx = dists.index(min(dists))
            return min_idx

        indexes = []
        for dist in np.arange(start_dist, start_dist + mark_count * dist_gap, dist_gap):
            line_at_dist = create_horizontal_line_at_depth(dist)
            indexes.append(get_idx_closest_point_to(lw, line_at_dist[0]))

        return indexes

    def add_3rd_dim(self, points):
        camera_position = self.camera_position

        points3d = []
        for point in points:
            points3d.append([
                point[0] + camera_position[0],
                0 + camera_position[1],
                point[1]]) # - camera_position[2]])

        return np.array(points3d)

    def render_steer(self, image, steer_angle):
        r = get_radius(steer_angle / WHEEL_STEER_RATIO)
        return self.render(image, r)

    def render(self, image, radius):
        mark_count = self.mark_count
        start_dist = self.start_dist
        gap_dist = self.gap_dist
        curve_length = self.curve_length
        center_color = self.center_color
        center_width = self.center_width
        side_color = self.side_color
        side_width = self.side_width

        indexes = self.detect_indexes_on_lane_points_for_distance_marks(mark_count,
                                                                        start_dist,
                                                                        gap_dist)

        c, lw, rw = get_car_path(radius, distance=curve_length)
        # print(lw[2:])
        c = self.add_3rd_dim(c)
        lw = self.add_3rd_dim(lw)
        rw = self.add_3rd_dim(rw)

        image = self.render_line(image, c, color=center_color, thickness=center_width)
        image = self.render_line(image, lw, color=side_color, thickness=side_width)
        image = self.render_line(image, rw, color=side_color, thickness=side_width)

        # for index in indexes:
        #     image = self.render_line(image, np.array([lw[index], rw[index]]), color=side_color,
        #                              thickness=side_width)
        # Render Distance lines
        for distance in self.distance_mark:
            c, lw, rw = get_car_line_mark(radius, distance)
            line = np.array([lw, rw])
            line = self.add_3rd_dim(line)
            # line = self.add_3rd_dim(np.array([[10, -1], [10, 1]]))
            # print(line)
            image = self.render_line(image, line, color=(0, 0, 255), thickness=side_width,
                                     filter_nonmonotonic=False, filter_negative=False)

        return image


def main_revised():

    cfg_i = DEFAULT_CFG

    cfg = Namespace()
    cfg.__dict__ = cfg_i

    tv = TrajectoryVisualizer(cfg)
    num = 400
    max_wheel_angle = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
    angles = np.linspace(-max_wheel_angle, max_wheel_angle, num)
    idx = int(angles.size / 2)
    r = -10.0

    data = {
        "r": r,
    }

    cap = cv2.VideoCapture(0) # 0 for built-in webcam, 1 for secondary, 2 for third ...
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame_from_image():
        rgb = cv2.imread("/media/nemodrive0/Samsung_T5/nemodrive/24_nov/session1/1543056917.67_camera_3_off_183ms.jpg")
        return rgb

    def get_frame_from_webcam():
        ret, rgb = cap.read()
        h, w = rgb.shape[:2]
        rgb[h//2:h//2 + 2, :] = np.array([0, 255, 128], dtype = np.uint8)
        rgb[:, w // 2:w // 2 + 2] = np.array([0, 255, 128], dtype=np.uint8)

        return rgb

    def loop():
        r = data['r']
        background_img = get_frame_from_image()
        #background_img = get_frame_from_webcam()
        # print(r)
        image = tv.render(background_img, r)
        image = cv2.resize(image, (1280, 720))
        cv2.imshow("image", image)

    def update(val):
        data['r'] = get_radius(angles[400 - val - 1])
        loop()

    def update_steer(val):
        data['r'] = get_radius((val-500) / WHEEL_STEER_RATIO)
        loop()

    cv2.namedWindow('image')
    # cv2.createTrackbar('angle', 'image', idx, 400 - 1, update)
    cv2.createTrackbar('angle', 'image', 0, 1000, update_steer)

    while True:
        loop()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main_revised()
