"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import math
import numpy as np
import cv2
from argparse import Namespace

FILTER_NEGATIVE_POINTS = True
FILTER_NONMONOTONIC_POINTS = True

CAR_L = 2.634  # Wheel base
CAR_T = 1.733  # Tread
MIN_TURNING_RADIUS = 5.
MAX_STEER = 500

CAMERA_POSITION = [0, 1.5422, 1.657]
CAMERA_MATRIX = [
    np.array([[1173.122620, 0.000000, 969.335924],
              [0.000000, 1179.612539, 549.524382],
              [0., 0., 1.]])
]
RVEC = np.array([0.04, 0.0, 0.0])
TVEC = np.array([0, 0, 0], np.float)  # translation vector
DISTORTION = np.array([0.053314, -0.117603, -0.004064, -0.001819, 0.000000])

CAMERA_HEIGHT = 1.56
cauciucuri_exterior = 1.69


def get_radius(angle, car_l=CAR_L, car_t=CAR_T):
    r = car_l / np.tan(np.deg2rad(angle, dtype=np.float32))
    return r


def get_delta(r, car_l=CAR_L, car_t=CAR_T):
    """
    :param r: Turn radius ( calculated against back center)
    :param car_l: Wheel base
    :param car_t: Tread
    :return: Angles of front center, inner wheel, outer wheel
    """
    delta_i = np.rad2deg(np.arctan(car_l / (r - car_t / 2.)))
    delta = np.rad2deg(np.arctan(car_l / r))
    delta_o = np.rad2deg(np.arctan(car_l / (r + car_t / 2.)))
    return delta, delta_i, delta_o


def get_car_path(r, distance=1., no_points=100, center_x=True, car_l=CAR_L, car_t=CAR_T):
    """
    :param r: Car turn radius ( against back center )
    :param distance: Distance to draw points
    :param no_points: No of points to draw on path
    :param center_x: If center point on the oX axis
    :param car_l: Wheel base
    :param car_t: Tread
    :return: center_points, inner_points, outer_points (on car path)
    """
    r_center = r
    r_inner = r_center - car_t / 2.
    r_outer = r_center + car_t / 2.

    d_inner = r_inner / r_center * distance
    d_outer = r_outer / r_center * distance
    center_points = point_on_circle(r_center, distance=distance, no_points=no_points,
                                    center_x=False)
    inner_points = point_on_circle(r_inner, distance=d_inner, no_points=no_points, center_x=False)
    outer_points = point_on_circle(r_outer, distance=d_outer, no_points=no_points, center_x=False)
    if center_x:
        center_points[:, 0] -= r_center
        inner_points[:, 0] -= r_center
        outer_points[:, 0] -= r_center

    return center_points, inner_points, outer_points


def point_on_circle(r, distance=1., no_points=100, center_x=True):
    """
    Returns a fix number of points on a circle circumference.
    :param r: circle radius
    :param distance: length of circumference to generate points for
    :param no_points: number of points to generate
    :param center: center points on the x axis ( - r)
    :return: np. array of 2D points
    """
    fc = 2 * np.pi * r
    p = distance / fc
    step = 2 * np.pi * p / float(no_points)
    points = np.array(
        [(math.cos(step * x) * r, math.sin(step * x) * r) for x in range(0, no_points + 1)]
    )
    if center_x:
        points[:, 0] = points[:, 0] - r
    return points


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
        self.rvec = cfg.RVEC
        self.center_color = cfg.center_color
        self.center_width = cfg.center_width
        self.side_color = cfg.side_color
        self.side_width = cfg.side_width
        self.curve_length = cfg.curve_length
        self.initial_steering = cfg.initial_steering
        self.RVEC = cfg.RVEC
        self.TVEC = cfg.TVEC
        self.CAMERA_MATRIX = cfg.CAMERA_MATRIX
        self.CAMERA_POSITION = cfg.CAMERA_POSITION
        self.DISTORTION = cfg.DISTORTION
        self.MARK_COUNT = cfg.MARK_COUNT
        self.START_DIST = cfg.START_DIST
        self.GAP_DIST = cfg.GAP_DIST
        self.CAMEAR_Y = cfg.CAMEAR_Y

    def project_points_on_image_space(self, points_3d):
        rvec = self.RVEC
        tvec = self.TVEC
        camera_matrix = self.CAMERA_MATRIX
        distortion = self.DISTORTION

        points = np.array(points_3d).astype(np.float32)
        points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
        points = points.astype(np.int)
        points = points.reshape(-1, 2)
        return points

    def render_line(self, image, points, color=None, thickness=None):
        points = self.project_points_on_image_space(points)

        h, w, _ = image.shape

        if FILTER_NEGATIVE_POINTS:
            idx = 0
            while (points[idx] < 0).any() or points[idx+1][1] < points[idx][1]:
                idx += 1
                if idx >= len(points) - 1:
                    break

            points = points[idx:]

        # TODO Check validity - expect monotonic modification on x
        # monotonic decrease on y
        if len(points) <= 0:
            return image

        prev_x, prev_y = points[0]
        idx = 1
        if len(points) > 1:
            while points[idx][1] < prev_y:
                prev_x, prev_y = points[idx]
                idx += 1
                if idx >= len(points):
                    break
                print(points[idx])

            valid_points = []
            while points[idx][0] >= 0 and points[idx][0] < w and idx < len(points)-1:
                valid_points.append([points[idx][0], points[idx][1]])
                idx += 1
        else:
            valid_points = [[prev_x, prev_y]]

        if FILTER_NONMONOTONIC_POINTS:
            points = np.array(valid_points)

        for p in zip(points, points[1:]):
            image = cv2.line(image, tuple(p[0]), tuple(p[1]), color=color, thickness=thickness)

        return image

    def detect_indexes_on_lane_points_for_distance_marks(self, mark_count, start_dist, dist_gap):
        initial_steering = self.initial_steering
        curve_length = self.curve_length

        c, lw, rw = get_car_path(initial_steering, distance=curve_length)
        lw = self.add_3rd_dim(lw)

        def create_horizontal_line_at_depth(distance_from_camera, left_limit=-CAR_T/2, right_limit=CAR_T/2, n=2):
            x = np.expand_dims(np.linspace(left_limit, right_limit, num=n), axis=1)
            y = np.ones((n, 1)) * CAMERA_HEIGHT
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
        camera_position = self.CAMERA_POSITION

        points3d = []
        for point in points:
            points3d.append([
                point[0] + camera_position[0],
                0 + camera_position[1],
                point[1]]) # - camera_position[2]])

        return np.array(points3d)

    def render(self, image, steering_angle):
        MARK_COUNT = self.MARK_COUNT
        START_DIST = self.START_DIST
        GAP_DIST = self.GAP_DIST
        curve_length = self.curve_length
        center_color = self.center_color
        center_width = self.center_width
        side_color = self.side_color
        side_width = self.side_width

        indexes = self.detect_indexes_on_lane_points_for_distance_marks(MARK_COUNT,
                                                                        START_DIST,
                                                                        GAP_DIST)

        c, lw, rw = get_car_path(steering_angle, distance=curve_length)
        c = self.add_3rd_dim(c)
        lw = self.add_3rd_dim(lw)
        rw = self.add_3rd_dim(rw)

        image = self.render_line(image, c, color=center_color, thickness=center_width)
        image = self.render_line(image, lw, color=side_color, thickness=side_width)
        image = self.render_line(image, rw, color=side_color, thickness=side_width)

        for index in indexes:
            image = self.render_line(image, np.array([lw[index], rw[index]]), color=side_color,
                                     thickness=side_width)

        return image


def main_revised():

    cfg_i = {'center_color': (0, 255, 0),
           'center_width': 4,
           'side_color': (0, 255, 255),
           'side_width': 2,
           'curve_length': 30.0,
           'initial_steering': -1994.999999999999971, # for this steering we have a straight line of trajectory

           'RVEC': RVEC,
           'TVEC': TVEC,
           'CAMERA_MATRIX': CAMERA_MATRIX[0],
           'CAMERA_POSITION': CAMERA_POSITION,
           'DISTORTION': DISTORTION,
           'MARK_COUNT': 10,
           'START_DIST': 6.0,
           'GAP_DIST': 1.0,
           'CAMEAR_Y': CAMERA_HEIGHT}
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
    #cap = cv2.VideoCapture('camera_3.mp4')
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame_from_image():
        rgb = cv2.imread("/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542535225.01_camera_1_off_7ms.jpg")
        # return cv2.resize(rgb, (1280, 720))
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
        data['r']  = get_radius(angles[400 - val - 1])
        loop()

    cv2.namedWindow('image')
    cv2.createTrackbar('angle', 'image', idx, 400 - 1, update)

    while True:
        loop()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main_revised()
