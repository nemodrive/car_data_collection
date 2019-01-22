"""
    Turn radius projection demo
    How and why to use the Ackermann steering model. https://www.youtube.com/watch?v=i6uBwudwA5o
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from datetime import datetime
import math

CAR_L = 2.634  # Wheel base - ampatament
CAR_T = 1.497  # Tread - ecartament fata vs 1.486 ecartament spate
MIN_TURNING_RADIUS = 5. # Seems ok, found specs with 5.25 or 5.5, though
# WHEEL_STEER_RATIO = 18.053225
WHEEL_STEER_RATIO = 18.0

# OFFSET_STEERING = 16.904771342679405
# OFFSET_STEERING = 16.394771342679383
# OFFSET_STEERING = 15.794771342679383
OFFSET_STEERING = 15.720720720720720
OFFSET_STEERING = 14.41051051051051
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


def get_car_line_mark(r, distance, center_x=True, car_t=CAR_T):
    center_point, inner_point, outer_point = get_car_path(r, distance, no_points=1,
                                                          center_x=center_x, car_t=car_t)
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


def get_car_can_path(speed_df, steer_df, steering_offset=OFFSET_STEERING, wheel_steer_ratio=WHEEL_STEER_RATIO):
    """ Approximate car coordinates from CAN info: speed & steer"""
    # speed_df, steer_df = speed.copy(), steer.copy()
    # ss = None

    speed_df = speed_df.sort_values("tp")
    steer_df = steer_df.sort_values("tp")

    # Update steer and speed (might be initialized correctly already)
    steer_df.steer = steer_df.can_steer + steering_offset
    speed_df["mps"] = speed_df.speed * 1000 / 3600.

    ss = pd.merge(speed_df, steer_df, how="outer", on=["tp"])
    ss = ss.sort_values("tp")

    # Make sure first row has values
    first_speed = speed_df.iloc[0]["mps"]
    first_steer = steer_df.iloc[0]["steer"]
    first_idx = ss.iloc[0].name
    ss.at[first_idx, "mps"] = first_speed
    ss.at[first_idx, "steer"] = first_steer

    # Time interpolation of steer and speed
    ss["rel_tp"] = ss.tp - ss.tp.min()
    ss["datetime"] = ss.tp.apply(datetime.fromtimestamp)
    ss = ss.set_index("datetime")
    ss.mps = ss.mps.interpolate(method="time")
    ss.steer = ss.steer.interpolate(method="time") * -1
    ss["radius"] = (ss.steer / wheel_steer_ratio).apply(get_radius)

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
    rel_move = np.column_stack([x2, y2])
    rel_move = np.vstack([[0, 0], car_offset[0], rel_move])

    cum_coord = np.cumsum(rel_move, axis=0)

    df_coord = pd.DataFrame(np.column_stack([rel_move, cum_coord, ss.tp.values]), columns=[
        "move_x", "move_y", "coord_x", "coord_y", "tp"])

    # fig = plt.figure()
    # plt.scatter(car_pos[:60000, 0], car_pos[:60000, 1], s=1.)
    # plt.axes().set_aspect('equal')

    return df_coord


def get_points_rotated(coord, orientation, offset_x, offset_y):
    omega = np.deg2rad(orientation)
    cos_o = np.cos(omega)
    sin_o = np.sin(omega)

    r = np.array([[cos_o, -sin_o], [sin_o, cos_o]])

    new_coord = np.dot(r, coord.transpose()).transpose()

    offset = np.array([offset_x, offset_y])
    new_coord = new_coord + offset

    return new_coord


def get_rotation_and_steering_offset(speed, steer, gps_unique_points,
                                     guess_orientation=180., guess_offest_x=0., guess_offest_y=0.,
                                     guess_steering_offset=OFFSET_STEERING,
                                     guess_wheel_steer_ratio=WHEEL_STEER_RATIO,
                                     maxiter=4000., tol=1e-10, fatol=1e-10, simple=False, idx=-1):
    import scipy.optimize as optimize

    gps_unique = gps_unique_points.copy()
    gps_unique["datetime"] = gps_unique.tp.apply(datetime.fromtimestamp)
    gps_unique = gps_unique.set_index("datetime")

    gps_unique.loc[:, "target_x"] = gps_unique.easting - gps_unique.iloc[0].easting
    gps_unique.loc[:, "target_y"] = gps_unique.northing - gps_unique.iloc[0].northing

    def fit_2d_curve(params):

        # WHEEL_STEER_RATIO OPTIM
        # orientation, offset_x, offset_y, wheel_steer_ratio = params
        # can_coord = get_car_can_path(speed, steer, wheel_steer_ratio=wheel_steer_ratio)

        # OFFSET_STEERING OPTIM
        orientation, offset_x, offset_y, steering_offset = params
        can_coord = get_car_can_path(speed, steer, steering_offset=steering_offset,
                                     wheel_steer_ratio=guess_wheel_steer_ratio)

        # ==========================================================================================
        # -- Can optimize code ... (operations that can be done not every curve fit)

        can_coord.loc[:, "datetime"] = can_coord.tp.apply(datetime.fromtimestamp).values
        df_coord = can_coord.set_index("datetime")

        nearest_car_pos = df_coord.reindex(gps_unique.index, method='nearest')

        merge_info = gps_unique.merge(nearest_car_pos, how="outer", left_index=True,
                                      right_index=True)

        coord = merge_info[["coord_x", "coord_y"]].values

        target = merge_info[["target_x", "target_y"]].values

        # ==========================================================================================

        omega = np.deg2rad(orientation)
        cos_o = np.cos(omega)
        sin_o = np.sin(omega)

        r = np.array([[cos_o, -sin_o], [sin_o, cos_o]])

        new_coord = np.dot(r, coord.transpose()).transpose()

        offset = np.array([offset_x, offset_y])
        new_coord = new_coord + offset

        diff = np.linalg.norm(new_coord - target, axis=1).sum() * 1000
        return diff

    # initial_guess = [guess_orientation, guess_offest_x, guess_offest_y, guess_wheel_steer_ratio]
    # if idx in [0, 1, 5, 8, 13, 16]:
    #     bnd_wheel_steel_ratio = (18., 18.)
    # else:
    #     bnd_wheel_steel_ratio = (17., 21.)
    # bnds = ((0., 350.), (-4., 4.), (-4., 4.), bnd_wheel_steel_ratio)

    initial_guess = [guess_orientation, guess_offest_x, guess_offest_y, guess_steering_offset]
    bnds = ((0., 350.), (-4., 4.), (-4., 4.), (14., 20.))

    if simple:
        result = optimize.minimize(fit_2d_curve, initial_guess, tol=tol, options={'maxiter': 1500})
    else:
        result = optimize.minimize(fit_2d_curve, initial_guess, method='Nelder-Mead', tol=tol,
                                   options={'maxiter': maxiter, "fatol": fatol})

    loss = fit_2d_curve(result["x"])
    result["loss"] = loss

    best_orientation, best_offest_x, best_offest_y, best_steering_offset = result["x"]
    best_wheel_steer_ratio = guess_wheel_steer_ratio

    # best_steering_offset = OFFSET_STEERING
    # best_orientation, best_offest_x, best_offest_y, best_wheel_steer_ratio = result["x"]

    df_coord = get_car_can_path(speed, steer, steering_offset=best_steering_offset,
                                wheel_steer_ratio=best_wheel_steer_ratio)

    all_coord = df_coord[["move_x", "move_y"]].values
    all_coord = np.cumsum(all_coord, axis=0)

    new_points = get_points_rotated(all_coord, *result.x[:3])
    new_points = pd.DataFrame(np.column_stack([new_points, df_coord.tp.values]),
                              columns=["coord_x", "coord_y", "tp"])

    return new_points, gps_unique, result


def get_rotation(df_coord, gps_unique_points, guess_orientation=180.,
                 guess_offest_x=0., guess_offest_y=0.,
                 maxiter=4000., tol=1e-10, fatol=1e-10, simple=True):
    """
    :param df_coord: Pandas dataframe with columns ["move_x", "move_y", "tp"]
    :param gps_data: Pandas dataframe with columns ["easting", "northing", "tp"]
    :param guess_orientation, guess_offest_x, guess_offest_y
    :return:
    """
    import scipy.optimize as optimize

    # Approximate course

    df_coord.loc[:, "datetime"] = df_coord.tp.apply(datetime.fromtimestamp).values
    df_coord = df_coord.set_index("datetime")

    # gps_unique = gps_data[
    #     (gps_data.tp >= df_coord.tp.min()) & (gps_data.tp <= df_coord.tp.max())
    # ].groupby(['loc_tp']).head(1)
    gps_unique = gps_unique_points
    gps_unique["datetime"] = gps_unique.tp.apply(datetime.fromtimestamp)
    gps_unique = gps_unique.set_index("datetime")

    nearest_car_pos = df_coord.reindex(gps_unique.index, method='nearest')

    # Filter out time interval
    # max_tp = gps_unique.tp.min() + 300.
    #
    # gps_unique = gps_unique[gps_unique.tp < max_tp]
    # nearest_car_pos = nearest_car_pos[nearest_car_pos.tp < max_tp]
    gps_unique.loc[:, "target_x"] = gps_unique.easting - gps_unique.iloc[0].easting
    gps_unique.loc[:, "target_y"] = gps_unique.northing - gps_unique.iloc[0].northing

    merge_info = gps_unique.merge(nearest_car_pos, how="outer", left_index=True, right_index=True)

    coord = merge_info[["coord_x", "coord_y"]].values

    # coord = merge_info[["move_x", "move_y"]].values
    # coord = np.cumsum(coord, axis=0)

    target = merge_info[["target_x", "target_y"]].values

    def fit_2d_curve(params):

        orientation, offset_x, offset_y = params

        omega = np.deg2rad(orientation)
        cos_o = np.cos(omega)
        sin_o = np.sin(omega)

        r = np.array([[cos_o, -sin_o], [sin_o, cos_o]])

        new_coord = np.dot(r, coord.transpose()).transpose()

        offset = np.array([offset_x, offset_y])
        new_coord = new_coord + offset

        diff = np.linalg.norm(new_coord - target, axis=1).sum()
        return diff

    # -------------
    initial_guess = [guess_orientation, guess_offest_x, guess_offest_y]

    result = optimize.minimize(fit_2d_curve, initial_guess )
    loss = fit_2d_curve(result["x"])
    result["loss"] = loss

    # result = optimize.minimize(fit_2d_curve, initial_guess, method='BFGS', tol=1e-15,
    #                            options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-14, 'maxiter': 4000})
    # loss = fit_2d_curve(result["x"])
    # print(loss)
    #
    # result = optimize.minimize(fit_2d_curve, initial_guess, method='Nelder-Mead', tol=1e-15,
    #                            options={'maxiter': 4000, "fatol": 1e-15})
    # loss = fit_2d_curve(result["x"])
    # print(loss)


    all_coord = df_coord[["move_x", "move_y"]].values
    all_coord = np.cumsum(all_coord, axis=0)

    new_points = get_points_rotated(all_coord, *result.x)
    new_points = pd.DataFrame(np.column_stack([new_points, df_coord.tp.values]),
                              columns=["coord_x", "coord_y", "tp"])
    #
    # fig = plt.figure()
    # plt.scatter(target[:, 0], target[:, 1], s=1.)
    # plt.axes().set_aspect('equal')
    #
    # # -------------
    #
    # fig = plt.figure()
    # plt.scatter(nearest_car_pos.move_x, nearest_car_pos.move_y, s=1.)
    # plt.axes().set_aspect('equal')
    #
    # fig = plt.figure()
    # plt.scatter(gps_unique.easting, gps_unique.northing, s=1.)
    # plt.axes().set_aspect('equal')

    return new_points, gps_unique, result


def get_car_path_orientation(phone, steer, speed, aprox_t_period=1.0, aprox_t_length=36.,
                             prev_t_factor=0.):

    first_tp, max_tp = phone.tp.min(), phone.tp.max()
    data_t_len = max_tp - first_tp

    starts = np.arange(0, data_t_len, aprox_t_period) + first_tp

    # Filter first still coord
    starts = starts[starts > speed[speed.speed > 0].iloc[0]["tp"]]
    ends = starts + aprox_t_length

    gps_data = phone.groupby(['loc_tp']).head(1)
    gps_data = gps_data[["easting", "northing", "tp"]]

    can_coord = get_car_can_path(speed, steer)

    all_results = []
    gps_splits = []
    for t_start, t_end in zip(starts, ends):
        gps_data_split = gps_data[(gps_data.tp >= t_start) & (gps_data.tp < t_end)]
        gps_splits.append(gps_data_split)
        can_coord_split = can_coord[(can_coord.tp >= t_start) & (can_coord.tp < t_end)]

        if len(gps_data_split) <= 0:
            print(f"No gps data in [{t_start}, {t_end}]")
            continue

        if len(can_coord_split) <= 0:
            print(f"No can_coord_split data in [{t_start}, {t_end}]")
            continue

        # can_coord[""]
        new_points = get_rotation(can_coord_split.copy(), gps_data_split.copy())
        all_results.append(new_points)

        if len(all_results) > 100:
            break

    idx = 0
    x = all_results[idx][0]

    fig = plt.figure()
    plt.plot(x["coord_x"], x["coord_y"])
    plt.axes().set_aspect('equal')

    gps = gps_splits[idx]
    fig = plt.figure()
    plt.scatter(gps.easting - gps.easting.min(), gps.northing - gps.northing.min())
    plt.axes().set_aspect('equal')


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




