"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import matplotlib.pyplot as plt
import math
import numpy as np

CAR_L = 2.634  # Wheel base
CAR_T = 1.733  # Tread
MIN_TURNING_RADIUS = 5.
MAX_STEER = 500


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
    points = np.array([
        (math.cos(step * x) * r, math.sin(step * x) * r) for x in range(0, no_points + 1)
    ])
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


if __name__ == "__main__":
    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()

    fig = plt.figure()

    car_l, car_t = CAR_L, CAR_T
    r = -10.
    c, lw, rw = get_car_path(r, distance=20)

    plt.plot(c[:, 0], c[:, 1])
    plt.plot(lw[:, 0], lw[:, 1])
    plt.plot(rw[:, 0], rw[:, 1])

    plt.axis('equal')
    plt.show(block=False)

    num = 400
    max_wheel_angle = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))

    angles = np.linspace(-max_wheel_angle, max_wheel_angle, num)

    idx = int(angles.size / 2)
    while True:
        fig.clear()
        fig.canvas.draw()
        fig.canvas.flush_events()

        r = get_radius(angles[idx])
        c, lw, rw = get_car_path(r, distance=20)
        print(c)

        plt.plot(c[:, 0], c[:, 1])
        plt.plot(lw[:, 0], lw[:, 1])
        plt.plot(rw[:, 0], rw[:, 1])

        plt.axis('equal')
        plt.show(block=False)

        q = raw_input("key:\n")
        if q == "q":
            idx -= 1
        elif q == "w":
            idx += 1




