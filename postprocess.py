import matplotlib.pyplot as plt
import cantools
import pandas as pd
import cv2
import numpy as np
import os
import glob
import re
import subprocess
import json

LOG_FOLDER = "/media/andrei/Samsung_T51/nemodrive/25_nov/session_2/1543155398_log"
CAN_FILE_PATH = os.path.join(LOG_FOLDER, "can_raw.log")
DBC_FILE = "logan.dbc"
SPEED_CAN_ID = "354"
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

db = cantools.database.load_file(DBC_FILE, strict=False)

cmd_names = [
    ("SPEED_SENSOR", "SPEED_KPS"),
    ("STEERING_SENSORS", "STEER_ANGLE"),
    ("BRAKE_SENSOR", "PRESIUNE_C_P")
]
cmd_idx = 0
cmd_name = cmd_names[cmd_idx][0]
data_name = cmd_names[cmd_idx][1]

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

df_can_speed = df_can[df_can["can_id"] == SPEED_CAN_ID]

df_can_speed["speed"] = df_can_speed["data_str"].apply(lambda x: get_can_data(db, cmd_name,
                                                                              data_name, x))

# df_can_speed[df_can_speed.speed > 0]
# df_can_speed["pts"] = df_can_speed["tp"] - 1539434950.220346

plt.plot(df_can_speed["tp"].values, df_can_speed["speed"])
plt.show()

# Write to csv
speed_file = os.path.join(LOG_FOLDER, "speed.csv")
df_can_speed.to_csv(speed_file)

# =======================================================================
# -- Load steer command

STEER_CMD_NAME = "STEERING_SENSORS"
STEER_CMD_DATA_NAME = "STEER_ANGLE"
STEERING_CAN_ID = "0C6"

df_can_steer = df_can[df_can["can_id"] == STEERING_CAN_ID]

df_can_steer["steering"] = df_can_steer["data_str"].apply(lambda x: get_can_data(db,
                                                                                 STEER_CMD_NAME,
                                                                                 STEER_CMD_DATA_NAME, x))

# Write to csv
steer_file = os.path.join(LOG_FOLDER, "steer.csv")
df_can_steer.to_csv(steer_file)

# --Plot can data
plt.plot(df_can_steer["tp"].values, df_can_steer["steering"])

plt.show()

# steering_values = []
# rng = 100
# for index in range(rng, len(df_can_steer)):
#     x = df_can_steer.iloc[index-rng: index+1]["steering"].values
#     steering_values.append(np.abs(x[1:] - x[:-1]).sum())
#
# steering_values_df = pd.Series(steering_values[:36494], name="Steering angle per second")
# steering_values_df.describe()
# # steering_values_df.plot()
# steering_values_df.plot(kind="box")
# plt.show()


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

df = phone_data_to_df(phone_log_path)
df.to_pickle(phone_log_path + ".pkl")

# ==================================================================================================
# -- CAMERA processing

# camera_start_tp = None
# with open(CAMERA_FILE_PREFIX + "_timestamp") as file:
#     data = file.read()
#     camera_start_tp = float(data)
#
# camera_start_tp = 1539434950.130855 - 35.4
#
# pts_file = pd.read_csv(CAMERA_FILE_PREFIX + "_pts.log", header=None)
# pts_file["tp"] = pts_file[0] + camera_start_tp
#
# # search for each frame the closest speed info
# camera_speed = pts_file.copy()
# camera_speed["speed"] = -1.0
# camera_speed["dif_tp"] = -333.
#
# pos_can = 0
#
# v = []
# prev = 0
# for index, row in pts_file.iterrows():
#     v.append(row[0] - prev)
#     prev = row[0]
#
#
# for index, row in camera_speed.iterrows():
#     frame_tp = row["tp"]
#     dif_crt = df_can_speed.iloc[pos_can]["tp"] - frame_tp
#     dif_next = df_can_speed.iloc[pos_can+1]["tp"] - frame_tp
#
#     while abs(dif_next) < abs(dif_crt) and pos_can < len(df_can_speed):
#         dif_crt = df_can_speed.iloc[pos_can]["tp"] - frame_tp
#         dif_next = df_can_speed.iloc[pos_can + 1]["tp"] - frame_tp
#         pos_can += 1
#
#     if pos_can >= len(df_can_speed):
#         print("reached end of df_can_speed")
#
#     row["speed"] = df_can_speed.iloc[pos_can]["speed"]
#     row["dif_tp"] = dif_crt
#
#
# # Analyze
# camera_speed["dif_tp"].describe()
#
# # Write camera speed file
# with open(CAMERA_FILE_PREFIX + "_speed.log", 'w') as filehandle:
#     for listitem in camera_speed["speed"].values:
#         filehandle.write('%.2f\n' % listitem)

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
video_frame_move = np.array([1444, 1411, 1438])

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

# vid = cv2.VideoCapture(CAMERA_FILE_PREFIX + ".mkv")
#
# for i in range(1410):
#     ret, frame = vid.read()
# i += 1
#
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# prev = frame
# while True:
#     i += 1
#     print(i)
#     ret, frame = vid.read()
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("Test", frame-prev)
#     cv2.imshow("Test", frame)
#     prev = frame
#     cv2.waitKey(0)
#
#
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# prev = frame
#
# while True:
#     i += 1
#     print(f"Frame:{i}_speed:{camera_speed.loc[i]['speed']}")
#     ret, frame = vid.read()
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # image = frame-prev
#     # show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     show = frame
#     cv2.putText(show, f"{camera_speed.loc[i]['speed']}", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 0, 200), 4)
#     cv2.imshow("Test", show)
#     prev = frame
#     cv2.waitKey(0)
#




