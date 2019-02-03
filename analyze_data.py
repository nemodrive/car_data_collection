import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from phone_data_utils import UPB_XLIM_EASTING, UPB_YLIM_NORTHING

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

CFG_FILE = "cfg.yaml"
CFG_EXTRA_FILE = "cfg_extra.yaml"

SPEED_FILE = "speed.csv"
STEER_FILE = "steer.csv"
PHONE_FILE = "phone.log.pkl"






def main():
    exp = "/media/nemodrive0/Samsung_T5/nemodrive/25_nov/session_2/1543155398_log"
    phone = pd.read_pickle("{}/phone.log.pkl".format(exp))
    steer = pd.read_csv("{}/steer.csv".format(exp))
    speed = pd.read_csv("{}/speed.csv".format(exp))

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
#  Plot Accelerometer

    acc = pd.DataFrame.from_dict(list(phone["acceleration"].values))
    acc["tp"] = phone["tp"]
    phone_start_tp = acc.tp.min()
    acc["tp_rel"] = acc.tp - phone_start_tp
    # acc["tp_rel"] = acc.tp - camer_start_time

    tp_start = 11*60+ 45. + 1542296336.71 - 30.
    acc_sel = acc[(acc.tp >= tp_start) & (acc.tp < tp_start + 30.)]

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(acc_sel.tp_rel, acc_sel.x)
    ax[1].plot(acc_sel.tp_rel, acc_sel.y)
    ax[2].plot(acc_sel.tp_rel, acc_sel.z)

    fig, ax = plt.subplots()
    ax.plot(acc.tp_rel, acc.z)
    ax.set_title("acc_z")

    fig, ax = plt.subplots()
    ax.plot(acc.tp_rel, acc.y)
    ax.set_title("acc_y")

    fig, ax = plt.subplots()
    ax.plot(acc.tp_rel, acc.x)
    ax.set_title("acc_x")


    speed["tp_rel"] = speed.tp - phone_start_tp
    fig, ax = plt.subplots()
    ax.plot(speed.tp_rel, speed.mps)
    ax.set_title("speed")
# ==================================================================================================
#  Plot Accelerometer

    userAcceleration = pd.DataFrame.from_dict(list(phone["userAcceleration"].values))
    userAcceleration["tp"] = phone["tp"]
    phone_start_tp = acc.tp.min()
    userAcceleration["tp_rel"] = userAcceleration.tp - phone_start_tp

    tp_start = 11*60+ 45. + 1542296336.71 - 30.
    acc_sel = acc[(acc.tp >= tp_start) & (acc.tp < tp_start + 30.)]


    fig, ax = plt.subplots()
    ax.plot(userAcceleration.tp_rel, userAcceleration.z)

# ==================================================================================================
#  Plot Accelerometer

    gravity = pd.DataFrame.from_dict(list(phone["gravity"].values))
    gravity["tp"] = phone["tp"]
    phone_start_tp = acc.tp.min()
    gravity["tp_rel"] = gravity.tp - phone_start_tp



    fig, ax = plt.subplots()
    ax.plot(gravity.tp_rel, gravity.z)
    ax.set_title("gravity")

# ==================================================================================================
#  Plot attiude
    attiude = pd.DataFrame.from_dict(list(phone["attitudeEularAngles"].values))
    attiude["tp"] = phone["tp"]
    phone_start_tp = phone.tp.min()
    attiude["tp_rel"] = attiude.tp - phone_start_tp

    camera_2_start = 1542296336.71
    tp_start = 450 + camera_2_start

    attiude_sel = attiude[(attiude.tp >= tp_start) & (attiude.tp < tp_start + 300.)]

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(attiude_sel.tp_rel, attiude_sel.x)
    ax[1].plot(attiude_sel.tp_rel, attiude_sel.y)
    ax[2].plot(attiude_sel.tp_rel, attiude_sel.z)

    fig, ax = plt.subplots()
    ax.plot(attiude.tp_rel, np.cumsum(attiude.y.values))


# ==================================================================================================
#  Interesting data segments
base_exp_path = "/media/nemodrive0/Samsung_T5/nemodrive"

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

phone_splits = []
steer_splits = []
speed_splits = []

for data_dict in DATA:
    experiment_path = data_dict["exp_path"]
    gps_unique_idx = data_dict["gps_unique_idx"]

    phone = pd.read_pickle("{}/phone.log.pkl".format(experiment_path))
    steer = pd.read_csv("{}/steer.csv".format(experiment_path))
    speed = pd.read_csv("{}/speed.csv".format(experiment_path))

    gps_unique = phone.groupby(['loc_tp']).head(1)

    for i, j in gps_unique_idx:
        tp_start, tp_end = gps_unique.iloc[i]["tp"], gps_unique.iloc[j]["tp"]

        phone_split = phone[(phone.tp >= tp_start) & (phone.tp < tp_end)]
        steer_split = steer[(steer.tp >= tp_start) & (steer.tp < tp_end)]
        speed_split = speed[(speed.tp >= tp_start) & (speed.tp < tp_end)]

        phone_splits.append(phone_split)
        steer_splits.append(steer_split)
        speed_splits.append(speed_split)

LOOPS = [
    {
        "exp_path": f"{base_exp_path}/18_nov/session_0/1542537659_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            # ["00:00.0", "10:46.12"], # 2 loops
            ["02:43.18", "06:47.10"],
            ["06:36.02", "10:31.19"],
            ["11:17.26", "15:00.9"],
            ["14:58.1", "18:29.26"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/18_nov/session_1/1542549716_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["05:40.29", "09:26.15"],
            ["19:21.26", "21:12.21"],
            ["22:06.27", "23:09.23"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/24_nov/session0/1543059528_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["01:27.5", "07:04.16"],
            ["06:38.6", "12:13.29"],
            ["23:49.6", "27:45.2"],
        ]
    }
]

STRAIGHT_LINE = [
    {
        "exp_path": f"{base_exp_path}/15_nov/1542296320_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["22:21.27", "22:50.08"],
            ["23:37.24", "23:43.17"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/18_nov/session_0/1542537659_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["06:09.24", "06:38.23"],
            ["08:33.14", "08:45.19"],
            ["10:07.20", "10:30.26"],
            ["11:20.30", "11:48.25"],
            ["15:00.11", "15:24.20"],
            ["20:59.11", "20:14.03"],
            ["21:38.13", "21:57.29"],
            ["22:16.11", "22:32.20"],
            ["23:07.21", "23:19.21"],
            ["23:46.08", "24:01.14"],
            ["24:22.23", "24:29.09"],
            ["24:44.06", "24:59.12"],
            ["26:26.18", "26:40.25"],
            ["30:52.01", "31:10.23"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/18_nov/session_1/1542549716_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["03:09.09", "03:17.29"],
            ["04:28.06", "04:48.20"],
            ["05:41.29", "06:10.13"],
            ["07:28.25", "07:36.11"],
            ["12:03.07", "12:20.06"],
            ["12:38.23", "12:48.20"],
            ["13:02.27", "13:07.12"],
            ["13:20.25", "13:38.13"],
            ["14:33.05", "14:38.08"],
            ["16:05.14", "16:16.20"],
            ["16:33.05", "16:43.12"],
            ["17:52.06", "18:08.11"],
            ["18:53.14", "19:03.07"],
            ["21:00.25", "22:03.26"],
            ["23:14.13", "23:30.3"],
            ["23:57.5", "24:06.27"],
            ["24:21.24", "24:37.29"],
            ["27:23.21", "27:49.29"],
            ["30:50.16", "31:02.18"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/24_nov/session0/1543059528_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["02:10.13", "02:45.25"],
            ["11:44.4", "12:01.01"],
            ["15:20.13", "15:31.19"],
            ["18:56.28", "19:15.01"],
            ["19:42.13", "19:56.22"],
            ["20:32.16", "20:43.01"],
            ["22:05.25", "22:16.04"],
            ["22:05.25", "22:16.04"],
        ]
    },
    {
        "exp_path": f"{base_exp_path}/25_nov/session_0/1543134132_log",
        "reference_time":"camera_2_timestamp",
        "tp": [
            ["03:10.13", "03:17.10"],
            ["03:29.25", "03:41.07"],
            ["05:03.25", "05:18.01"],
            ["06:11.13", "06:37.07"],
            ["10:14.22", "10:43.25"],
            ["12:10.07", "12:26.04"],
            ["12:50.28", "13:08.04"],
            ["14:38.25", "14:46.25"],
            ["17:08.10", "17:18.01"],
            ["17:30.22", "17:45.25"],
            ["18:50.19", "19:01.25"],
            ["19:10.4", "19:20.22"],
            ["21:58.4", "22:10.01"],
            ["22:30.19", "22:40.13"],
            ["23:37.19", "23:52.28"],
            ["24:05.28", "24:20.28"],
        ]
    },
]


def parse_video_time_format(s, fps=30.):
    """ Format MM:SS.f """
    m, sf = s.split(":")
    m = float(m)
    s, f = [float(x) for x in sf.split(".")]

    time_interval = m * 60. + s + 1. /fps * f
    return time_interval


# # SHOULD use margins for save cut FOR STRAIGHT LINE
# margin_right = 1.0
# margin_left = 1.

# # SHOULD use margins for save cut
margin_right = 0.0
margin_left = 0.0

phone_splits = []
steer_splits = []
speed_splits = []

for data_dict in LOOPS:  # STRAIG
    experiment_path = data_dict["exp_path"]

    phone = pd.read_pickle("{}/phone.log.pkl".format(experiment_path))
    steer = pd.read_csv("{}/steer.csv".format(experiment_path))
    speed = pd.read_csv("{}/speed.csv".format(experiment_path))
    reference_time = data_dict['reference_time']
    time_intervals = data_dict["tp"]

    with open(f"{experiment_path}/{reference_time}") as f:
        camera_start_tp = float(f.read())

    gps_unique = phone.groupby(['loc_tp']).head(1)

    for i, j in time_intervals:
        tp_start = parse_video_time_format(i) + camera_start_tp
        tp_end = parse_video_time_format(j) + camera_start_tp

        phone_split = phone[(phone.tp >= tp_start + margin_left) & (phone.tp < tp_end - margin_right)]
        steer_split = steer[(steer.tp >= tp_start + margin_left) & (steer.tp < tp_end - margin_right)]
        speed_split = speed[(speed.tp >= tp_start + margin_left) & (speed.tp < tp_end - margin_right)]

        phone_splits.append(phone_split)
        steer_splits.append(steer_split)
        speed_splits.append(speed_split)


# ==============================================================================================
# Determine offset_steering ON STRAIGHT LINE

for idx, (phone_s, steer_s, speed_s) in enumerate(zip(phone_splits, steer_splits, speed_splits)):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.plot(gps_unique_points.easting, gps_unique_points.northing, c="r", zorder=1)
    # ax.plot(steer.tp, steer.can_steer, c="r", zorder=1)
    # fig.suptitle(f"{idx}_{len(gps_unique_points)}")
    # fig.savefig(f"data/Min_off_steering_steer_ratio_{idx}.png")
    plt.show()
    plt.waitforbuttonpress()
    plt.close("all")

for idx, steer in enumerate(steer_splits):
    steer["dataset_idx"] = idx

all_steer: pd.DataFrame = pd.concat(steer_splits)

all_steer.boxplot(column=["can_steer"], by="dataset_idx")

outliers = []
all_steer_wo_o = all_steer[all_steer.dataset_idx.apply(lambda x: x not in outliers)]

all_steer_wo_o.can_steer.plot.box()
all_steer_wo_o.can_steer.median()  # -15.4
all_steer_wo_o.can_steer.mean()  # -15.149503459604759
all_steer_wo_o.can_steer.describe()
# count    80356.000000
# mean       -15.149503
# std          3.221953
# min        -36.200000
# 25%        -16.800000
# 50%        -15.400000
# 75%        -13.600000
# max          1.200000
# Name: can_steer, dtype: float64

# ==============================================================================================
# Determine offset_steering by optimization

from car_utils import get_rotation_and_steering_offset
from multiprocessing import Pool as ThreadPool

all_res = []

no+=1

def get_rotation_and_steering_offset_mt(args):
    speed, steer, gps_unique_points, task = args
    print(f"Start {task}...")
    result = get_rotation_and_steering_offset(speed, steer, gps_unique_points, simple=False, idx=task)
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

# ==============================================================================================
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
# ==============================================================================================

# save data to disk
fld = "/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/optimize_with_ratio/optim_offset_ratio_18"
steer_optimization = dict({"DATA": DATA, "results": results, "args": args})
np.save(f"{fld}/steer_optimization_{no}.npy", steer_optimization)

# # Load from disk
# steer_optimization = np.load(f"{fld}/steer_optimization.npy").item()
# DATA = steer_optimization["DATA"]
# results = steer_optimization["results"]
# args = steer_optimization["args"]
#
# Run plot
steer_offset_res = []

nrows = 5
ncols = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

for idx, (new_points, gps_unique, result_opti) in enumerate(results):
    ax = axes[idx//ncols, idx%ncols]

    ax.scatter(new_points.coord_x, new_points.coord_y, s=1., c="r")

    ax.scatter(gps_unique.target_x, gps_unique.target_y, s=1., c="b")
    ax.set_aspect('equal')

    # best_orientation, best_offest_x, best_offest_y, best_steering_offset, best_wheel_steer_ratio = result_opti["x"]
    best_orientation, best_offest_x, best_offest_y, best_steering_offset = result_opti["x"]
    # best_steering_offset = OFFSET_STEERING
    ax.set_title(f"i:{idx} - s:{best_steering_offset}")

    steer_offset_res.append(best_steering_offset)

fig.set_size_inches(18.55, 9.86)
fig.savefig(f"{fld}/calc_wheel_ratio_{no}.png")

# res_info = pd.DataFrame([[*r[2]["x"], r[2]["success"], r[2]["message"], r[2]["nit"]] for r in results],
#                                 columns=["oritentaion", "offset_x", "offset_y", "wheel_steer_ratio",
#                                          "success", "meessage", "nit"])
res_info = pd.DataFrame([[*r[2]["x"], r[2]["success"], r[2]["message"], r[2]["nit"]] for r in results],
                                columns=["oritentaion", "offset_x", "offset_y", "steering_offset",
                                         "success", "meessage", "nit"])
res_info.to_csv(f"{fld}/optimization_info.csv",
                float_format = '%.14f')

all_res.append(res_info)

# Other stupid

fig = plt.figure()
res_info.steering_offset.plot(kind="bar")
fig.suptitle("Calculated steerings by optimization", fontsize=16)

# MERge
res_info pd.concat(all_res)
res_info = res_info.reset_index()
res_info = res_info.rename_axis({"index":"idx"}, axis=1)

# Wheel ration analysis
ignore_wheel = [0, 1, 5, 6, 8, 13, 16, 17, 18, 19]
wheel_steer_ratio_s = res_info[[x not in ignore_wheel for x in res_info["idx"]]].wheel_steer_ratio

fig = plt.figure()
wheel_steer_ratio_s.plot()

wheel_steer_ratio_s.describe()
# count    40.000000
# mean     18.053225
# std       0.812695
# min      16.955807
# 25%      17.350352
# 50%      17.947226
# 75%      18.581726
# max      19.416291

# Try different steering offsets
from car_utils import get_rotation, get_car_can_path

out_path = ""
maybe_offsets = np.linspace(14.555555555555555555 -0.7, 14.555555555555555555 + 0.7, 1000)
nrows = 5
ncols = 4

out_folder = "/media/nemodrive3/Samsung_T5/nemodrive/optimized_steering_offset/small_variation_18_0"
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
# Minimize loop closer

WHEEL_STEER_RATIO = 18.0
OFFSET_STEERING = 15.45000001

losses = []
var_offset_steering = [15.149503459604759, 15.400000000010011]
var_steer_ratio = np.linspace(17.8, 18.3, 100)

# MULTI PROCESS
from multiprocessing import Pool as ThreadPool
from car_utils import get_car_can_path
import time

def get_losses(args):
    speed, steer, dataset_idx = args
    losses = []
    print(f"Start {dataset_idx}...")
    for offset_steering in var_offset_steering:
        stp = time.time()

        for i, steer_ratio in enumerate(var_steer_ratio):
            can_coord = get_car_can_path(speed.copy(), steer.copy(),
                                         steering_offset=offset_steering, wheel_steer_ratio=steer_ratio)

            loss = np.linalg.norm(can_coord.iloc[-1][["coord_x", "coord_y"]].values)
            losses.append([loss, steer_ratio, offset_steering, dataset_idx])
        print(f"Done {dataset_idx}: {i} ({time.time()-stp})!")
    return losses

threads = 9
pool = ThreadPool(threads)
args = []
for idx, (phone_s, steer_s, speed_s) in enumerate(zip(phone_splits, steer_splits, speed_splits)):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    args.append((speed, steer, idx))

results = pool.map(get_losses, args)
pool.close()
pool.join()

r = []
for x in results:
    r += x

from mpl_toolkits.mplot3d import Axes3D

losses = pd.DataFrame(r, columns=["loss", "steer_ratio", "offset_steering", "dataset_idx"])

# Plot all datasets
fig = plt.figure()
for idx in range(len(phone_splits)):
    df = losses[losses.dataset_idx == idx]
    ax = fig.add_subplot(3, 4, idx+1, projection='3d')
    ax.scatter(df.steer_ratio, df.offset_steering, df.loss, s=1.)
fig.suptitle("Grid search steer offset and ratio for loops")

# Plot minimum loss coord
all_data_loss = losses.groupby(["steer_ratio", "offset_steering"])["loss"].apply(lambda x: x.pow(4, axis=0).sum())
all_data_loss = all_data_loss.reset_index()
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(all_data_loss.steer_ratio, all_data_loss.offset_steering, all_data_loss.loss, s=1.)

min_off_s, min_s_r = all_data_loss.loc[all_data_loss.loss.idxmin()][["offset_steering", "steer_ratio"]].values


# SEEMS like median is the best !
# All steering smaller than 18. seems weird !

selection = losses[losses.offset_steering == var_offset_steering[1]]
min_loss = selection.groupby("dataset_idx").loss.idxmin()
selection = selection.loc[min_loss]

min_steer_ratio = selection[selection.steer_ratio > 18.0].steer_ratio.mean()

matplotlib.rcParams['figure.figsize'] = (18.55, 9.86)

min_off_s  = var_offset_steering[1]
min_s_r = min_steer_ratio
"""
    BEST RESULT: 
        OFFSET: 15.40000000001001
         RATIO: 18.215151515151515

"""
# Plot minimum loss coord
for idx, (phone_s, steer_s, speed_s) in enumerate(zip(phone_splits, steer_splits, speed_splits)):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)

    # df = losses[losses.dataset_idx == idx]
    # min_idx = df.loss.idxmin()
    # min_conf = df.loc[min_idx]
    # #
    # min_off_s = min_conf.offset_steering
    # min_s_r = min_conf.steer_ratio
    # print(f"IDX: {idx} min_ {min_off_s} _ {min_s_r}")
    # can_coord = get_car_can_path(speed.copy(), steer.copy(),
    #                              steering_offset=min_off_s, wheel_steer_ratio=min_s_r)
    can_coord = get_car_can_path(speed.copy(), steer.copy(),
                                 steering_offset=min_off_s, wheel_steer_ratio=min_s_r)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(can_coord.coord_x, can_coord.coord_y, s=1.5, c="r", zorder=1)
    ax[0].set_aspect('equal')
    ax[1].scatter(gps_unique_points.easting, gps_unique_points.northing, s=1.5, c="r", zorder=1)
    ax[1].scatter(gps_unique_points.iloc[0].easting, gps_unique_points.iloc[0].northing, s=1.5, c="b", zorder=1)
    ax[1].scatter(gps_unique_points.iloc[-1].easting, gps_unique_points.iloc[-1].northing, s=1.5, c="black", zorder=1)
    ax[1].set_aspect('equal')
    fig.suptitle(f"Offset_steering: {min_off_s}. Steer ratio: {min_s_r}")
    fig.savefig(f"data/Min_off_steering_steer_ratio_{idx}.png")
    plt.show()
    plt.waitforbuttonpress()
    plt.close("all")


# -- Iterative
dataset_idx = 0


for phone_s, steer_s, speed_s in zip(phone_splits, steer_splits, speed_splits):
    phone, steer, speed = phone_s.copy(), steer_s.copy(), speed_s.copy()
    gps_unique_points = phone.groupby(['loc_tp']).head(1)

    for steer_ratio in var_steer_ratio:
        for offset_steering in var_offset_steering:

            can_coord = get_car_can_path(speed.copy(), steer.copy(),
                                         steering_offset=offset_steering, wheel_steer_ratio=steer_ratio)

            loss = np.linalg.norm(can_coord.iloc[-1][["coord_x", "coord_y"]].values)
            losses.append([loss, steer_ratio, offset_steering, dataset_idx])
            # fig = plt.figure()
            # plt.scatter(can_coord.coord_x, can_coord.coord_y, s=1.5, c="r", zorder=1)
            # plt.axes().set_aspect('equal')
            # plt.show()

    dataset_idx += 1

gps = gps_unique[gps_unique.tp < tp]
fig = plt.figure()
plt.plot(gps.easting - gps.easting.min(), gps.northing -
         gps.northing.min(), zorder=-1)
plt.scatter(gps.easting - gps.easting.min(), gps.northing -
            gps.northing.min(), s=1.5, c="r", zorder=1)
plt.show()
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


#



