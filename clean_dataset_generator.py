"""
    Info.json format

"""
import pandas as pd
import datetime
import json
import os
import re
import uuid
from datetime import datetime
import subprocess
from subprocess import PIPE
import numpy as np

from utils import namespace_to_dict, exclude_intervals
from analyze_data import load_experiment_data, adjust_camera_cfg

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

UPB_XLIM_EASTING = [423772., 424808.]  # recorded [423792.118788, 424788.577689]
UPB_YLIM_NORTHING = [4920534., 4921623.]  # recorded [4920554.23386, 4921603.97505]


INFO_FILE = "info.json"
SEGMENT_SIZE = 36.  # Seconds

# If True -> smaller segments will be appended to previous segments (if available)
STRICT_SIZE = False
VIDEO_SIZE = [640, 360]  # Image out size Width height

FFMPEG_CMD = "ffmpeg -i {} -s {}x{} -ss {} -t {} -vcodec libx264 {}.mov"

TIME_TO_MS = True

GROUPS = dict({
    # Group name : {1: [<regex>, ] #include, 0: [<regex>, ] #exclude}
    "good": {1: ["full"], 0: ["bad_*", "long_stationary"]}
})


def run_video_cut_cmd(video_path, video_size, start_time, duration, out_path):
    print(f"Run video cut cmd: {video_path} :: {start_time} :: len({duration}) ...")

    ffmpeg_start_time = ffmpeg_time_format(start_time)
    ffmpeg_duration = ffmpeg_time_format(duration)
    cmd = FFMPEG_CMD.format(video_path, video_size[0], video_size[1],
                            ffmpeg_start_time, ffmpeg_duration, out_path)

    pro = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, bufsize=1, shell=True,
                           preexec_fn=os.setsid)
    return pro


def ffmpeg_time_format(seconds: float):
    return datetime.utcfromtimestamp(seconds).strftime("%H:%M:%S.%f")


def get_bdd_like_locations(phone, speed, steer, get_time=None):
    print("Generate BDD like locations intervals ...")

    gps_unique = phone.groupby(['loc_tp']).head(1)

    # Get necessary columns from phone data
    columns = dict({
        "trueHeading": "course",
        "tp": "timestamp",
        "latitude": "latitude",
        "longitude": "longitude",
        "loc_accuracy": "accuracy",
    })
    location = gps_unique[list(columns.keys())].copy()

    # Get only loc accuracy on x
    location.loc[:, "loc_accuracy"] = location.loc_accuracy.apply(lambda x: x["x"])

    # Get closest speed from CAN DATA
    speed = speed.sort_values("tp")
    first_speed = speed.iloc[0]["mps"]
    ss = pd.merge(location, speed, how="outer", on=["tp"])
    ss = ss.sort_values("tp")
    ss["datetime"] = ss.tp.apply(datetime.fromtimestamp)
    ss = ss.set_index("datetime")
    first_idx = ss.iloc[0].name
    ss.at[first_idx, "mps"] = first_speed
    ss.mps = ss.mps.interpolate(method="time")

    location["speed"] = ss.loc[location.tp.apply(datetime.fromtimestamp)]["mps"].values

    # Fix timestamp
    if get_time is None:
        get_time = lambda x: x

    location.tp = location.tp.apply(get_time)

    # Correct column names
    location = location.rename(columns, axis=1)

    return location


def filter_phone_data(phone):
    phone = phone.drop(["location", "msg_client_ids", "zone_no", "zone_letter", "global",
                        "updateInterval"], axis=1)
    return phone


def filter_speed_data(speed):
    speed = speed.drop(["Unnamed: 0", "0", "1", "2", "can_id", "data_str", "speed"], axis=1)
    return speed


def filter_steer_data(steer):
    steer = steer.drop(["Unnamed: 0", "0", "1", "2", "can_id", "data_str"], axis=1)
    return steer


def main(experiment_path, out_fld, export_videos=True):

    # Read info
    with open(os.path.join(experiment_path, INFO_FILE)) as f:
        info = json.load(f)

    get_time = lambda x: x
    if TIME_TO_MS:
        get_time = lambda x: x * 1000.

    segments = info["segments"]
    time_reference = info["time_reference"]

    # Load experiment utils
    print("Start loading experiment data ...")
    edata = load_experiment_data(experiment_path)
    print("Loaded experiment data...")

    # Filter big data
    edata.phone = filter_phone_data(edata.phone)
    edata.speed = filter_speed_data(edata.speed)
    edata.steer = filter_steer_data(edata.steer)

    # get time reference
    tp_reference = [x.start_timestamp for x in edata.cameras if x.name == time_reference][0]

    for group, group_rules in GROUPS.items():
        # Build group valid intervals
        valid_ = []
        exclude_ = []

        for rule in group_rules[1]:
            for k, v in segments.items():
                if re.match(rule, k) is not None:
                    valid_ += v
        for rule in group_rules[0]:
            for k, v in segments.items():
                if re.match(rule, k) is not None:
                    exclude_ += v

        valid_intervals = exclude_intervals(valid_, exclude_)
        print(f"Valid {time_reference} intervals: {valid_intervals}")

        cut_intervals = []
        for start, end in valid_intervals:
            no_segm = (end - start) // SEGMENT_SIZE
            segm = np.linspace(start, start + no_segm * SEGMENT_SIZE, no_segm + 1)
            segm[-1] = end

            if len(segm) > 1:
                cut_intervals += list(zip(segm[:-1], segm[1:]))

        # Generate group folder
        g_out_fld = os.path.join(out_fld, group)
        assert not os.path.isdir(g_out_fld), f"Group folder exists {g_out_fld}"
        os.mkdir(g_out_fld)

        # Generate BDD Like locations
        locations = get_bdd_like_locations(edata.phone, edata.speed, edata.steer, get_time=get_time)

        for start, end in cut_intervals:
            print(f"Generate interval data: {start} - {end}")
            procs = []

            data = dict()
            data["rideID"] = re.sub('-', "", f"{uuid.uuid4()}")

            start_time = tp_reference + start
            end_time = tp_reference + end
            data["startTime"] = start_time_ms = get_time(tp_reference + start)
            data["endTime"] = end_time_ms = get_time(tp_reference + end)

            # -- CAMERA STUFF

            # Start video cutting
            data["cameras"] = []
            camera_base_path = os.path.join(g_out_fld, data["rideID"][:16])
            for _id, camera in enumerate(edata.cameras):
                offset = tp_reference - camera.start_timestamp
                if offset >= 0:
                    if export_videos:
                        procs.append(run_video_cut_cmd(camera.video_path, VIDEO_SIZE,
                                                       start + offset, end - start + offset,
                                                       f"{camera_base_path}-{_id}"))
                    new_cfg = namespace_to_dict(adjust_camera_cfg(camera, start + offset,
                                                                  end + offset))

                    # TODO Must calculate real pts of frames -> after conversion :(
                    new_cfg.pop("pts")

                    data["cameras"].append(new_cfg)
                else:
                    print(f"Camera Bad offset {camera.name}")

            # -- Phone STUFF
            # Write locations like bdd data
            data["locations"] = locations[
                (locations.timestamp >= start_time_ms) & (locations.timestamp <= end_time_ms)
            ].to_dict("records")

            # Write other PHONE data ....
            data["phone_data"] = edata.phone[
                (edata.phone.tp >= start_time) & (edata.phone.tp <= end_time)
            ].to_dict("list")
            print(len(data["phone_data"]["acceleration"]))

            # -- CAN STUFF
            # Write OTHER STEER data ....
            data["steer_data"] = edata.steer[
                (edata.steer.tp >= start_time) & (edata.steer.tp <= end_time)
            ].to_dict("list")

            # Write OTHER STEER data ....
            data["speed_data"] = edata.speed[
                (edata.speed.tp >= start_time) & (edata.speed.tp <= end_time)
            ].to_dict("list")

            # Write json data
            print("Writing json data ...")
            with open(os.path.join(g_out_fld, data["rideID"][:16]+".json"), 'w') as outfile:
                json.dump(data, outfile, indent=2)

            # Wait for video cut to end
            print("Waiting to write video segments ...")
            exit_codes = [p.wait() for p in procs]
            print(f"Done: {exit_codes}")

        # TODO Convert phone to json

        # TODO convert speed to json

        # TODO convert steer to json


if __name__ == '__main__':
    experiment_path = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log"
    out_fld = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/segments"
    export_videos = True

    main(experiment_path, out_fld, export_videos=export_videos)






