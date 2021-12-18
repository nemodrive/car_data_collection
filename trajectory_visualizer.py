"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import math
import numpy as np
import cv2
import os
import torch
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

    def filter_points(self, image, points,
                    filter_negative=FILTER_NEGATIVE_POINTS,
                    filter_nonmonotonic=FILTER_NONMONOTONIC_POINTS):
        points = self.project_points_on_image_space(points)
        # print(points)

        h, w, _ = image.shape

        w_err = 50

        if filter_negative:
            idx = 0
            while (points[idx] < 0).any() or points[idx+1][1] < points[idx][1]:
                idx += 1
                if idx >= len(points) - 1:
                    break

            points = points[idx:]
        else:
            points = points[(points >= 0).all(axis=1)]

        prev_x, prev_y = points[0]
        idx = 1

        if len(points) > 1 and filter_nonmonotonic:
            while points[idx][1] < prev_y:
                prev_x, prev_y = points[idx]
                idx += 1
                if idx >= len(points):
                    break

            valid_points = []
            while -w_err <= points[idx][0] < w + w_err and idx < len(points)-1:
                valid_points.append([points[idx][0], points[idx][1]])
                idx += 1
        else:
            valid_points = [[prev_x, prev_y]]

        if filter_nonmonotonic:
            points = np.array(valid_points)

        if len(points) <= 0:
            return points


        points[:, 0] = np.clip(points[:, 0], 0, w)
        points[:, 1] = np.clip(points[:, 1], 0, h)

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

    def render_path(self, image, radius):
        curve_length = self.curve_length
        c, lw, rw = get_car_path(radius, distance=curve_length)
        lw = self.add_3rd_dim(lw)
        rw = self.add_3rd_dim(rw)

        lw = self.filter_points(image, lw)
        rw = self.filter_points(image, rw)

        overlay = np.zeros_like(image)

        for p1, p2, p3, p4 in zip(lw[:-1], rw[:-1], lw[1:], rw[1:]):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            pts = np.array([(x1, y1), (x3, y3), (x4, y4), (x2, y2)])

            overlay = cv2.drawContours(overlay, [pts], 0, (0, 255, 0), cv2.FILLED)

        image = cv2.addWeighted(image, 1, overlay, 1, 0)

        return image, overlay

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


def normalize(tensor):
    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]
    return tensor


def compute_score(overlay, predicted):
    return predicted[overlay != 0].mean()


def compute_miou(overlay, predicted):
    predicted = predicted / predicted.max()
    c1 = overlay.astype(bool)
    c2 = predicted.astype(bool)

    overlap = c1 * c2
    union = c1 + c2

    miou = overlap.sum() / float(union.sum())

    return miou


def main_revised():

    cfg_i = DEFAULT_CFG

    cfg = Namespace()
    cfg.__dict__ = cfg_i

    tv = TrajectoryVisualizer(cfg)
    num = 1001
    max_wheel_angle = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
    angles = np.linspace(-MAX_STEER, MAX_STEER, num)
    r = 1

    data = {
        "r": r,
    }

    def get_frame_from_image(path):
        rgb = cv2.imread(path)
        return rgb

    def loop(path):
        r = data['r']
        background_img = get_frame_from_image(path)
        image, overlay = tv.render_path(background_img, r)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        return image, overlay

    def update_steer(path, camera_matrix, val):
        tv.camera_matrix = camera_matrix
        data['r'] = get_radius(val / WHEEL_STEER_RATIO)
        return loop(path)

    test_dir = '/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/test_frames/'
    soft_labels_dir = '/home/nemodrive/workspace/andreim/awesome-semantic-segmentation-pytorch/runs/pred_tensor_all/deeplabv3_resnet50_upb/'
    hard_labels_dir = '/home/nemodrive/workspace/andreim/awesome-semantic-segmentation-pytorch/runs/pred_pic_hard_all/deeplabv3_resnet50_upb/'

    save_dir_soft = '/home/nemodrive/workspace/andreim/self_supervised_steering_results/soft'
    save_dir_hard = '/home/nemodrive/workspace/andreim/self_supervised_steering_results/hard'

    test_dirs = os.listdir(test_dir)

    for d in sorted(test_dirs):
        if '.txt' not in d:
            dir_path = os.path.join(test_dir, d)
            test_files = os.listdir(dir_path)
            cam_file = os.path.join(dir_path, 'cam.txt')
            camera_matrix = np.loadtxt(cam_file)
            for f in sorted(test_files):
                print(f)
                if '.txt' not in f:
                    file_path = os.path.join(dir_path, f)

                    sequence_frame = '/'.join(file_path.split('/')[-2:])

                    sequence_frame_path_soft = os.path.join(soft_labels_dir, sequence_frame).replace('.png', '.pt')
                    sequence_frame_path_hard = os.path.join(hard_labels_dir, sequence_frame)

                    sequence_frame_soft = normalize(torch.load(sequence_frame_path_soft)).cpu().numpy()
                    sequence_frame_hard = cv2.imread(sequence_frame_path_hard)[:, :, 1]

                    frame_results_soft = []
                    frame_results_hard = []

                    for angle in angles:
                        image, overlay = update_steer(file_path, camera_matrix, angle)
                        image = cv2.addWeighted(image, 1, np.repeat(sequence_frame_hard[:, :, None], 3, axis=2), 1, 0)

                        # cv2.imshow('image', image)
                        # cv2.waitKey(0)

                        overlay = overlay[:, :, 1]
                        overlay = overlay / overlay.max()

                        soft_score = compute_score(overlay, sequence_frame_soft)
                        miou = compute_miou(overlay, sequence_frame_hard)

                        frame_results_soft.append(soft_score)
                        frame_results_hard.append(miou)

                    res_soft = np.array([angles, frame_results_soft])
                    res_hard = np.array([angles, frame_results_hard])

                    np.save(os.path.join(save_dir_soft, sequence_frame).replace('.png', '.npy'), res_soft)
                    np.save(os.path.join(save_dir_hard, sequence_frame).replace('.png', '.npy'), res_hard)

if __name__ == "__main__":
    main_revised()
