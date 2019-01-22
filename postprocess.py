import matplotlib.pyplot as plt
import cantools
import pandas as pd
import numpy as np
import os
import glob
import re
import subprocess

LOG_FOLDER = "/media/nemodrive0/Samsung_T5/nemodrive/15_nov/1542296320_log"
CAN_FILE_PATH = os.path.join(LOG_FOLDER, "can_raw.log")
OBD_SPEED_FILE = LOG_FOLDER + "obd_SPEED.log"
CAMERA_FILE_PREFIX = os.path.join(LOG_FOLDER, "camera_*")

camera_vids_path = glob.glob(CAMERA_FILE_PREFIX + ".mkv")
camera_vids_path.sort()
vid_names = [os.path.splitext(os.path.basename(vid_path))[0] for vid_path in camera_vids_path]
vid_dirs = [os.path.dirname(vid_path) for vid_path in camera_vids_path]

cameras = [os.path.join(x, y) for x, y in zip(vid_dirs, vid_names)]
camera_logs_path = [camera_name + ".log" for camera_name in cameras]
cameras_tp_path = [camera_name + "_timestamp" for camera_name in cameras]

phone_log_path = os.path.join(LOG_FOLDER, "phone.log")


def read_can_file(can_file_path):
    df_can = pd.read_csv(can_file_path, header=None, delimiter=" ")
    df_can["tp"] = df_can[0].apply(lambda x: float(x[1:-1]))
    df_can["can_id"] = df_can[2].apply(lambda x: x[:x.find("#")])
    df_can["data_str"] = df_can[2].apply(lambda x: x[x.find("#") + 1:])
    return df_can


def get_can_data(db, cmd, data, msg):
    decoded_info = db.decode_message(cmd, bytearray.fromhex(msg))
    return decoded_info[data]


# =======================================================================
# extract pts data

PTS_CMD = "ffprobe -v error -show_entries frame=pkt_pts_" \
          "time -of default=noprint_wrappers=1:nokey=1 {} > {}_pts.log"


for vid_path, vid_name in zip(camera_vids_path, cameras):
    cmd = PTS_CMD.format(vid_path, vid_name)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)


# =======================================================================
# -- Load DBC file stuff
from can_utils import DBC_FILE, CMD_NAMES

db = cantools.database.load_file(DBC_FILE, strict=False)


"""
    Decode values using: 
    Define: cmd_name, data_name, raw_can_msg
    
    decoded_info = db.decode_message(cmd_name, bytearray.fromhex(raw_can_msg))
    print(decoded_info[data_name])
ffmpeg -f v4l2 -video_size {}x{} -i /dev/video{} -c copy {}.mkv
"""

# =======================================================================
# -- Load Raw can
df_can = read_can_file(CAN_FILE_PATH)

# =======================================================================
# -- Load speed command
cmd_name, data_name, can_id = CMD_NAMES["speed"]
df_can_speed = df_can[df_can["can_id"] == can_id]
df_can_speed["speed"] = df_can_speed["data_str"].apply(lambda x: get_can_data(db, cmd_name,
                                                                              data_name, x))

df_can_speed["mps"] = df_can_speed.speed * 1000 / 3600.

# Write to csv
speed_file = os.path.join(LOG_FOLDER, "speed.csv")
df_can_speed.to_csv(speed_file, float_format='%.6f')

plt.plot(df_can_speed["tp"].values, df_can_speed["speed"])
plt.show()

# =======================================================================
# -- Load steer command
from car_utils import OFFSET_STEERING

cmd_name, data_name, can_id = CMD_NAMES["steer"]

df_can_steer = df_can[df_can["can_id"] == can_id]
df_can_steer["steer"] = df_can_steer["data_str"].apply(lambda x: get_can_data(db, cmd_name,
                                                                              data_name, x))

# Make default correction
df_can_steer["can_steer"] = df_can_steer["steer"]
df_can_steer["steer"] = df_can_steer["steer"] + OFFSET_STEERING

# Write to csv
steer_file = os.path.join(LOG_FOLDER, "steer.csv")
df_can_steer.to_csv(steer_file, float_format='%.6f')

# --Plot can data
plt.plot(df_can_steer["tp"].values, df_can_steer["steer"])
plt.show()


# =======================================================================
# -- OBD speed file
df_speed = pd.read_csv(OBD_SPEED_FILE, header=None)
df_speed["value"] = df_speed[1].apply(lambda x: None if x is None else x.split()[0])
df_speed["value"] = df_speed["value"].apply(lambda x: None if x == "None" else float(x))
df_speed.set_index(0, inplace=True)
no_unique_val = df_speed["value"].nunique()

# ==================================================================================================
# -- PHONE processing
from phone_data_utils import phone_data_to_df
from phone_data_utils import get_gps

df = phone_data_to_df(phone_log_path)
gps = get_gps(df)
merged_data = pd.merge(df, gps, how="outer")

merged_data.to_pickle(phone_log_path + ".pkl")

# ==================================================================================================
# -- CAMERA processing


# ==================================================================================================
# -- Sync video movement # video setup

video_start_pts = []
for camera_log in camera_logs_path:

    with open(camera_log, 'r') as f:
        while True:
            line = f.readline()
            m = re.findall("Duration: N\/A, start: (.*), bitrate: N\/A", line)
            if len(m) > 0:
                break
        vsp = float(m[0])

        video_start_pts.append(vsp)
video_start_pts = np.array(video_start_pts)

# Approximate which is the first frame where the car moves
video_frame_move = np.array([9597, 9572, 9637])

# Extract pts of frame
pts_df = [
    pd.read_csv(os.path.join(vid_dir, vid_name + "_pts.log"), header=None)
    for vid_dir, vid_name in zip(vid_dirs, vid_names)]

video_frame_move_pts = np.array([j.loc[i, 0] for i, j in zip(video_frame_move, pts_df)])

video_pts_start_move = (video_start_pts + video_frame_move_pts).mean()
print("Deviation: {}".format(video_pts_start_move - (video_start_pts + video_frame_move_pts)))

# get the first CAN MSG tp where the speed is greater than 0
can_first_move_tp = df_can_speed[df_can_speed.speed > 0].iloc[0]["tp"]

video_start_tps = video_start_pts - video_pts_start_move + can_first_move_tp
for video_start_tp, camera_name in zip(video_start_tps, cameras):
    with open(camera_name + "_timestamp", "w") as f:
        f.write(str(video_start_tp))

# ==================================================================================================
# Get info about recorded intervals timpestamps

# CAN Interval data
min_max_can = [df_can.tp.min(), df_can.tp.max()]

# Cameras
camera_start = []
for camera_tp_file in cameras_tp_path:
    with open(camera_tp_file, "r") as f:
        camera_start.append(float(f.read()))

pts_df = [pd.read_csv(os.path.join(vid_dir, vid_name + "_pts.log"), header=None)
          for vid_dir, vid_name in zip(vid_dirs, vid_names)]

min_max_cameras = []
for i in range(len(camera_start)):
    min_max_cameras.append([camera_start[i], camera_start[i] + pts_df[i][0].iloc[-1]])


# Phone log
def get_tp_line(line):
    return float(line.split(";")[0])


phone_log = open(phone_log_path, "r")
phone_lines = phone_log.readlines()

min_max_phone = [get_tp_line(phone_lines[0]), get_tp_line(phone_lines[-1])]

intervals = np.array([min_max_can] + min_max_cameras + [min_max_phone])

# Plot
for i in range(len(intervals)):
    plt.plot(intervals[i], [i, i])

plt.show()

# Extract min start
recorded_min_max = [intervals.min(), intervals.max()]
common_min_max = [intervals[:, 0].max(), intervals[:, 1].min()]

intervals = intervals.tolist() + [recorded_min_max, common_min_max]
df_tp = pd.DataFrame(intervals,
                  index=["can"] + vid_names + ["phone", "recorded", "common"],
                  columns=["min", "max"])

for idx, row in df_tp.iterrows():
    print("{}_min_max_tp: [{}, {}]".format(idx, row["min"], row["max"]))

# ==================================================================================================
# -- Video others (random stuff)





