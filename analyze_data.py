import pandas as pd
import matplotlib.pyplot as plt
from utils import read_cfg
import os
from argparse import Namespace
import numpy as np
import copy

from phone_data_utils import UPB_XLIM_EASTING, UPB_YLIM_NORTHING

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

CFG_FILE = "cfg.yaml"
CFG_EXTRA_FILE = "cfg_extra.yaml"

SPEED_FILE = "speed.csv"
STEER_FILE = "steer.csv"
PHONE_FILE = "phone.log.pkl"



def adjust_camera_cfg(cfg, start=None, end=None):
    new_cfg = copy.deepcopy(cfg)

    # Most of them can be ignored
    delattr(new_cfg, "cfg")

    # Cut pts
    # Does not work :(
    # pts = np.array(new_cfg.pts)
    # pts_start = (pts >= start).argmax()
    # pts_end = (pts <= end).argmin()
    # new_cfg.pts = pts[pts_start:pts_end].tolist()
    #
    # Adjust extrinsic calibration
    extra_cfgs = new_cfg.cfg_extra
    if len(extra_cfgs) == 1:
        new_cfg.cfg_extra = extra_cfgs[0]
    else:
        for extra_cfg in extra_cfgs:
            valid_interval = getattr(extra_cfg, "valid_interval", [0, 0])
            if valid_interval[0] <= start and valid_interval[1] >= end:
                new_cfg.cfg_extra = extra_cfg
                break

    return new_cfg


def load_experiment_data(experiment_path):
    d = Namespace()

    # Read experiment CFG
    d.cfg = read_cfg(os.path.join(experiment_path, CFG_FILE))
    d.collect = d.cfg.collect
    d.record_timestamp = d.cfg.recorded_min_max_tp
    d.common_min_max_tp = d.cfg.common_min_max_tp

    # Read Phone data
    d.phone_log_path = os.path.join(experiment_path, PHONE_FILE)
    d.phone = pd.read_pickle(d.phone_log_path) if os.path.isfile(d.phone_log_path) else None

    # Read Can data
    d.steer_log_path = os.path.join(experiment_path, STEER_FILE)
    d.steer = pd.read_csv(d.steer_log_path) if os.path.isfile(d.steer_log_path) else None

    d.speed_log_path = os.path.join(experiment_path, SPEED_FILE)
    d.speed = pd.read_csv(d.speed_log_path) if os.path.isfile(d.speed_log_path) else None

    d.cfg_extra = read_cfg(os.path.join(experiment_path, CFG_EXTRA_FILE))

    # Read camera stuff
    d.cameras = []
    for camera in ["camera_{}".format(x) for x in d.cfg.camera.ids]:
        c = Namespace()
        c.name = camera
        c.video_path = os.path.join(experiment_path, "{}.mkv".format(camera))
        c.cfg = getattr(d.cfg.camera, camera)

        cfg_extras = [k for k in d.cfg_extra.__dict__.keys() if camera in k]
        c.cfg_extra = [getattr(d.cfg_extra, x) for x in cfg_extras]
        with open(os.path.join(experiment_path, f"{camera}_timestamp"), "r") as f:
            c.start_timestamp = float(f.read())

        c.pts = pd.read_csv(os.path.join(experiment_path, "{}_pts.log".format(camera)), header=None)
        c.pts.sort_index(inplace=True, ascending=True)
        c.pts = c.pts[0].values.tolist()
        d.cameras.append(c)

        # TODO Add camera start move frame

    return d


def main():
    exp = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_1/1542549716_log"
    phone = pd.read_pickle("{}/phone.log.pkl".format(exp))
    steer = pd.read_csv("{}/steer.csv".format(exp))
    speed = pd.read_csv("{}/speed.csv".format(exp))
    steer["can_steer"] = steer.steer

    offset_tp = 1543059543.52

    max_tp = df.loc[26800].tp
    df = df[df.tp < max_tp]
    speed = speed[speed.tp < max_tp]
    steer = steer[steer.tp < max_tp]

    with open("/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/segments/good/a417768c24bc483c.json") as f:
        info = json.load(f)
# ==================================================================================================
# Plot gps

    gps_unique = phone.groupby(['loc_tp']).head(1)

    fig = plt.figure()
    plt.plot(gps_unique.easting - gps_unique.easting.min(), gps_unique.northing -
                gps_unique.northing.min(), zorder=-1)
    plt.scatter(gps_unique.easting - gps_unique.easting.min(), gps_unique.northing -
                gps_unique.northing.min(), s=1.5, c="r", zorder=1)
    plt.show()
    plt.axes().set_aspect('equal')

# ==================================================================================================
#  Plot steering

    fig = plt.figure()
    plt.scatter(steer.tp, steer.steer, s=3.5)
    plt.show()


# ==================================================================================================
#  Plot Magnetometer

    fig = plt.figure()
    plt.plot(phone.tp - offset_tp, phone.trueHeading)

    fig = plt.figure()
    plt.plot(steer.tp - offset_tp, steer.steer)


# ==================================================================================================
#  Interesting data segments
base_exp_path = "/media/andrei/Samsung_T51/nemodrive"

DATA = [
    {
        "exp_path": f"{base_exp_path}/24_nov/session0/1543059528_log",
        "gps_unique_idx": [
            [57, 92],
            [89, 103],
            # [342, 371],  # MAYBE NOT THAT STRAIGHT
            [604, 625],
            [782, 820],
            [1217, 1229],
            [1335, 1356]
        ]
    },
    {
        "exp_path": f"{base_exp_path}/18_nov/session_1/1542549716_log",
        "gps_unique_idx": [
            [149, 158],
            [202, 215],
            [207, 246],
            [242, 265],
            [258, 287],
            [304, 332],
            [322, 341],
            [663, 700],
            [680, 700],
            [692, 729],
            [723, 747],
            [743, 780],
            [926, 959],
            [1214, 1284],
            [1279, 1362],  # Large loop closure
            [1660, 1717],  # Parallel road driving with full turn
            [1777, 1795],
            # [2144, 2194],  # Bad driving on straight road
            # [2216, 2243],  # Circle around round "island"
            # [2269, 2319],  # Max steering full circle turn
        ]
    }
]

for data_dict in DATA:
    experiment_path = data_dict["exp_path"]
    gps_unique_idx = data_dict["gps_unique_idx"]

    phone = pd.read_pickle("{}/phone.log.pkl".format(experiment_path))
    steer = pd.read_csv("{}/steer.csv".format(experiment_path))
    speed = pd.read_csv("{}/speed.csv".format(experiment_path))

    gps_unique = phone.groupby(['loc_tp']).head(1)

    phone_splits = []
    steer_splits = []
    speed_splits = []
    for i, j in gps_unique_idx:
        tp_start, tp_end = gps_unique.iloc[i]["tp"], gps_unique.iloc[j]["tp"]

        phone_split = phone[(phone.tp >= tp_start) & (phone.tp < tp_end)]
        steer_split = steer[(steer.tp >= tp_start) & (steer.tp < tp_end)]
        speed_split = speed[(speed.tp >= tp_start) & (speed.tp < tp_end)]

        phone_splits.append(phone_split)
        steer_splits.append(steer_split)
        speed_splits.append(speed_split)


# ==============================================================================================
# Determine offset_steering by optimization

from car_utils import get_rotation_and_steering_offset

results = []
i = 3
phone_s, steer_s, speed_s = phone_splits[3], steer_splits[3], speed_splits[3]
for phone_s, steer_s, speed_s in zip(phone_splits, steer_splits, speed_splits):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)
    result = get_rotation_and_steering_offset(speed, steer, gps_unique_points)

    results.append(result)
    i += 1
    print(f"Done {i}/{len(phone_splits)}")

new_points, gps_unique, result_opti  = result

fig = plt.figure()
plt.scatter(new_points.coord_x, new_points.coord_y, s=1.)
plt.axes().set_aspect('equal')

fig = plt.figure()
plt.scatter(gps_unique.target_x, gps_unique.target_y, s=1.)
plt.axes().set_aspect('equal')

# ==============================================================================================
# Plot GPS

results = []
for phone_s, steer_s, speed_s in zip(phone_splits, steer_splits, speed_splits):

    gps_unique = phone_s.groupby(['loc_tp']).head(1)

    fig = plt.figure()
    plt.plot(gps_unique.easting, gps_unique.northing, zorder=-1)
    plt.scatter(gps_unique.easting, gps_unique.northing, s=1.5, c="r", zorder=1)
    plt.xlim(UPB_XLIM_EASTING)
    plt.ylim(UPB_YLIM_NORTHING)
    plt.axes().set_aspect('equal')
    plt.show()
    plt.pause(0.0001)
    k = input()






