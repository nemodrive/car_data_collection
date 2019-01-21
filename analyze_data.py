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
base_exp_path = "/media/nemodrive3/Samsung_T5/nemodrive"

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
            [2144, 2194],  # Bad driving on straight road
            [2216, 2243],  # Circle around round "island"
            [2269, 2319],  # Max steering full circle turn
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
from multiprocessing import Pool as ThreadPool

def get_rotation_and_steering_offset_mt(args):
    speed, steer, gps_unique_points, task = args
    print(f"Start {task}...")
    result = get_rotation_and_steering_offset(speed, steer, gps_unique_points)
    print(f"Done {task}!")
    return result

threads = 11
pool = ThreadPool(threads)
args = []
for idx, (phone_s, steer_s, speed_s) in enumerate(zip(phone_splits, steer_splits, speed_splits)):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)

    args.append((speed, steer, gps_unique_points, idx))

results = pool.map(get_rotation_and_steering_offset_mt, args)
pool.close()
pool.join()

# Iterative version
results = []
# i = 3
# phone_s, steer_s, speed_s = phone_splits[3], steer_splits[3], speed_splits[3]
for phone_s, steer_s, speed_s in zip(phone_splits, steer_splits, speed_splits):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)
    result = get_rotation_and_steering_offset(speed, steer, gps_unique_points)

    results.append(result)
    i += 1
    print(f"Done {i}/{len(phone_splits)}")

# save data to disk
steer_optimization = dict({"DATA": DATA, "results": results, "args": args})
np.save("/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/steer_optimization.npy", steer_optimization)

steer_optimization = np.load("/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/steer_optimization.npy").item()
DATA = steer_optimization["DATA"]
results = steer_optimization["results"]
args = steer_optimization["args"]
steer_offset_res = []

nrows = 5
ncols = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

for idx, (new_points, gps_unique, result_opti) in enumerate(results):
    ax = axes[idx//ncols, idx%ncols]

    ax.scatter(new_points.coord_x, new_points.coord_y, s=1., c="r")

    ax.scatter(gps_unique.target_x, gps_unique.target_y, s=1., c="b")
    ax.set_aspect('equal')


    best_orientation, best_offest_x, best_offest_y, best_steering_offset = result_opti["x"]
    ax.set_title(f"i:{idx} - s:{best_steering_offset}")

    steer_offset_res.append(best_steering_offset)


res_info = pd.DataFrame([[*r[2]["x"], r[2]["success"], r[2]["message"], r[2]["nit"]] for r in results],
                                columns=["oritentaion", "offset_x", "offset_y", "steering_offset", "success",
                                         "meessage", "nit"])
res_info.to_csv("/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/optimization_info.csv",
                float_format = '%.14f')

fig = plt.figure()
res_info.steering_offset.plot(kind="bar")
fig.suptitle("Calculated steerings by optimization", fontsize=16)

# Try different steering offsets
from car_utils import get_rotation, get_car_can_path

out_path = ""
maybe_offsets = np.linspace(15.71921921921922 - 0.3, 15.71921921921922 + 0.3, 1000)
nrows = 5
ncols = 4

out_folder = "/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/small_variation"
full_results = []

for offset_sol in maybe_offsets:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    save_data = []
    for idx, (_a, _r) in enumerate(zip(args, results)):
        speed, steer, gps_unique, task = _a
        best_orientation, best_offest_x, best_offest_y, best_steering_offset = _r[2]["x"]

        can_coord = get_car_can_path(speed.copy(), steer.copy(), steering_offset=offset_sol)

        new_points, gps_unique, result = get_rotation(can_coord.copy(), gps_unique.copy(),
                                                      guess_orientation=best_orientation,
                                                      guess_offest_x=best_offest_x, guess_offest_y=best_offest_y)

        ax = axes[idx // ncols, idx % ncols]

        ax.scatter(new_points.coord_x, new_points.coord_y, s=1., c="r")

        ax.scatter(gps_unique.target_x, gps_unique.target_y, s=1., c="b")
        ax.set_aspect('equal')

        ax.set_title(f"i:{idx}")

        save_info = dict({"new_points": new_points, "gps_unique": new_points, "result": new_points})
        save_data.append(save_info)

        full_results.append([offset_sol, idx, result["nit"], result["success"], result["message"],
                             *result["x"], result["loss"]])

    fig.suptitle(f"Steering offset {offset_sol}")
    fig.set_size_inches(18.55, 9.86)
    fig.savefig(f"{out_folder}/offset_{offset_sol}.png")
    plt.close('all')

    np.save(f"{out_folder}/offset_{offset_sol}.npy", save_data)

full_results_df = pd.DataFrame(full_results, columns=["offset_sol", "idx", "nit", "success", "message", "orientation",
                                                      "offset_x", "offset_y", "loss"])
full_results_df.to_pickle(f"{out_folder}/full_results_df.pkl")

df = full_results_df[full_results_df.idx < 18]
df.groupby("offset_sol")["loss"].sum().plot()
losses = df.groupby("offset_sol")["loss"].sum()
losses.sort_values(ascending=True).head(10)

# maybe_offsets = np.linspace(15.71921921921922 - 0.3, 15.71921921921922 + 0.3, 1000)

# Minimum result 15.72072072072072
# 15.800000       0.000000 # Clearly
# 16.000000       0.000000
# 15.600000       0.000000
# 15.844444    2298.341425 # Anomaly
# 15.533333    2427.975330 # Anomaly
# 15.720721    2483.125663
# 15.720120    2483.125678
# 15.721321    2483.125689
# 15.719520    2483.125709
# 15.722523    2483.125715


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






