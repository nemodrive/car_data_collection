"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

CAR_L = 2.634  # Wheel base - ampatament
CAR_T = 1.497  # Tread - ecartament fata vs 1.486 ecartament spate
MIN_TURNING_RADIUS = 5. # Seems ok, found specs with 5.25 or 5.5, though
WHEEL_STEER_RATIO = 20.

# OFFSET_STEERING = 16.904771342679405
# OFFSET_STEERING = 16.394771342679383
OFFSET_STEERING = 15.794771342679383
# Best so far - record MAX can wheel turn -> diff with 5.m radius


def get_radius(wheel_angle, car_l=CAR_L):
    r = car_l / np.tan(np.deg2rad(wheel_angle, dtype=np.float64))
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


def get_car_path(r, distance=1., no_points=100, center_x=True, car_t=CAR_T):
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
    center_points = points_on_circle(r_center, distance=distance, no_points=no_points,
                                     center_x=False)
    inner_points = points_on_circle(r_inner, distance=d_inner, no_points=no_points, center_x=False)
    outer_points = points_on_circle(r_outer, distance=d_outer, no_points=no_points, center_x=False)
    if center_x:
        center_points[:, 0] -= r_center
        inner_points[:, 0] -= r_center
        outer_points[:, 0] -= r_center

    return center_points, inner_points, outer_points


def get_car_line_mark(r, distance, center_x=True, car_l=CAR_L, car_t=CAR_T):
    center_point, inner_point, outer_point = get_car_path(r, distance, no_points=1,
                                                          center_x=center_x,
                                                          car_l=car_l, car_t=car_t)
    return center_point[1], inner_point[1], outer_point[1]


def points_on_circle(r, distance=1., no_points=100, center_x=True):
    """
    Returns a fix number of points on a circle circumference.
    :param r: circle radius
    :param distance: length of circumference to generate points for
    :param no_points: number of points to generate
    :param center: center points on the x axis ( - r)
    :return: np. array of 2D points
    """
    fc = r
    p = distance / fc
    step = p / float(no_points)
    points = np.array([
        (math.cos(step * x) * r, math.sin(step * x) * r) for x in range(0, no_points + 1)
    ])
    if center_x:
        points[:, 0] = points[:, 0] - r
    return points


def get_car_offset(r, arc_length, center_x=False):
    """
    arc_len = r * Omega # omega angle in radians
    http://mathcentral.uregina.ca/QQ/database/QQ.09.07/s/bruce1.html

    :param r: circle radius
    :param distance: length of circumference to generate points for
    :param center: center points on the x axis ( - r)
    :return: np. array of 2D points
    """
    angle = arc_length / r
    x_offset = r - r * math.cos(angle)
    y_offset = r * math.sin(angle)
    point = np.array([x_offset, y_offset])
    if center_x:
        point[0] = point[0] - r
    return point


def get_car_can_pat(speed_df, steer_df):
    """ Approximate car coordinates from CAN info: speed & steer"""
    import datetime

    speed_df = speed_df.sort_values("tp")
    steer_df = steer_df.sort_values("tp")

    # Update steer and speed (might be initialized correctly already)
    speed_df.steer = steer_df.can_steer + OFFSET_STEERING
    speed_df["mps"] = speed_df.speed * 1000 / 3600.

    ss = pd.merge(speed_df, steer_df, how="outer", on=["tp"])
    ss = ss.sort_values("tp")

    # Make sure first row has values
    first_speed = speed_df.iloc[0]["mps"]
    first_steer = steer_df.iloc[0]["steer"]
    first_idx = ss.iloc[0].name
    ss.set_value(first_idx, "mps", first_speed)
    ss.set_value(first_idx, "steer", first_steer)

    # Time interpolation of steer and speed
    ss["rel_tp"] = ss.tp - ss.tp.min()
    ss["datetime"] = ss.tp.apply(datetime.fromtimestamp)
    ss = ss.set_index("datetime")
    ss.mps = ss.mps.interpolate(method="time")
    ss.steer = ss.steer.interpolate(method="time") * -1
    ss.radius = (ss.steer / WHEEL_STEER_RATIO).apply(get_radius)

    dist = (ss.mps[1:].values + ss.mps[:-1].values) / 2. * \
           (ss.rel_tp[1:].values - ss.rel_tp[:-1].values)
    r = ss.radius.values[1:]
    omega = dist / r
    omega = -omega.cumsum()[:-1]

    assert not np.isnan(dist).any(), "Got NaN values when calculating <dist?"
    assert not np.isnan(r).any(), "Got NaN values when calculating <r>"
    assert not np.isnan(omega).any(), "Got NaN values when calculating <omega>"

    data = np.column_stack([r, dist])
    car_offset = [get_car_offset(_r, _d) for _r, _d in data]
    car_offset = np.array(car_offset)

    cos_angle = np.cos(omega)
    sin_angle = np.sin(omega)

    x = car_offset[1:, 0]
    y = car_offset[1:, 1]

    x2 = cos_angle * x - sin_angle * y
    y2 = sin_angle * x + cos_angle * y
    car_pos = np.column_stack([x2, y2])
    car_pos = np.vstack([[0, 0], car_offset[0], car_pos])

    car_pos = np.cumsum(car_pos, axis=0)
    return car_pos


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

    c, lw, rw = get_car_line_mark(r, distance=20)

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

        q = input("key:\n")
        if q == "q":
            idx -= 1
        elif q == "w":
            idx += 1




