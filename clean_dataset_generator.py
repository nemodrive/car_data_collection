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
import time

from utils import namespace_to_dict, exclude_intervals
from data_utils import load_experiment_data, adjust_camera_cfg

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

FFMPEG_CMD = "ffmpeg -nostdin -i {} -s {}x{} -ss {} -t {} -vcodec libx264 {}"

TIME_TO_MS = True

GROUPS = dict({
    # Group name : {1: [<regex>, ] #include, 0: [<regex>, ] #exclude}
    "good": {1: ["full"], 0: ["bad_*", "long_stationary", "turn"]}
})

MAX_VIDEO_CUT_THREADS = 10
WAIT_TIME = 5.0


def run_video_cut_cmd(video_path, video_size, start_time, duration, out_path):
    print(f"Run video cut cmd: {video_path} :: {start_time} :: len({duration}) ...")

    ffmpeg_start_time = ffmpeg_time_format(start_time)
    ffmpeg_duration = ffmpeg_time_format(duration)
    cmd = FFMPEG_CMD.format(video_path, video_size[0], video_size[1],
                            ffmpeg_start_time, ffmpeg_duration, out_path)

    out_file = open(f"{out_path}.log", "w")

    pro = subprocess.Popen(cmd, stdout=out_file, stderr=out_file, stdin=PIPE, shell=True)
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


# ======================================================================================================================
# Usefull scripts

def croll_info(base_path):
    import glob
    info_paths = glob.glob(f"{base_path}/**/info.json", recursive=True)

    infos = []
    for x in info_paths:
        with open(x) as f:
            print(x)
            infos.append(json.load(f))

    segments_k = set()
    for x in infos:
        segments_k.update(x["segments"].keys())


# ======================================================================================================================
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
                    print(k)
                    valid_ += v
        for rule in group_rules[0]:
            for k, v in segments.items():
                if re.match(rule, k) is not None:
                    print(k)
                    # Change TP reference for bad_can & bad_phone to camera reference time
                    if k in ["bad_phone", "bad_can"]:
                        v = [[x[0] - tp_reference, x[1] - tp_reference] for x in v]

                    exclude_ += v
        valid_intervals = exclude_intervals(valid_, exclude_)
        cut_intervals = []
        for start, end in valid_intervals:
            no_segm = (end - start) // SEGMENT_SIZE
            segm = np.linspace(start, start + no_segm * SEGMENT_SIZE, no_segm + 1)

            # Try to extend penultimate interval (sometimes important stuff in the last interval)
            if len(segm) >= 3:
                extra = end - segm[-1]
                segm[-2:] += extra
            else:
                segm[-1] = end

            if len(segm) > 1:
                cut_intervals += list(zip(segm[:-1], segm[1:]))

        # Generate group folder
        g_out_fld = os.path.join(out_fld, group)
        if os.path.isdir(g_out_fld):
            print(f"Group folder exists {g_out_fld}")
            # k = input("Do you want to continue? y/n")
            # if k != "y":
            #     print("Continue to next group")
            #     continue
            continue_session = True
        else:
            continue_session = False
            os.mkdir(g_out_fld)

        # Generate BDD Like locations
        locations = get_bdd_like_locations(edata.phone, edata.speed, edata.steer, get_time=get_time)

        # Save export info
        export_info_path = os.path.join(g_out_fld, "export_info.npy")
        intervals_out = os.path.join(g_out_fld, "intervals.csv")

        if continue_session:
            export_info = np.load(export_info_path).item()
            cut_intervals = export_info["cut_intervals"]
            info = export_info["info"]
            tp_reference = export_info["tp_reference"]
            rideIDs = export_info["rideIDs"]
        else:
            intervals_out_f = open(intervals_out, "w")
            rideIDs = [re.sub('-', "", f"{uuid.uuid4()}") for _ in range(len(cut_intervals))]

            export_info = dict({"cut_intervals": cut_intervals, "experiment_path": experiment_path, "info": info,
                                "tp_reference": tp_reference, "groups": GROUPS, "g_out_fld": g_out_fld,
                                "rideIDs": rideIDs})

            np.save(export_info_path, export_info)

        video_cut_pocs = []

        for interval_no, (start, end) in enumerate(cut_intervals):
            print(f"Generate interval data: {start} - {end}")

            data = dict()
            data["rideID"] = rideID = rideIDs[interval_no]

            start_time = tp_reference + start
            end_time = tp_reference + end
            data["startTime"] = start_time_ms = get_time(tp_reference + start)
            data["endTime"] = end_time_ms = get_time(tp_reference + end)

            # -- CAMERA STUFF

            # Start video cutting
            data["cameras"] = []
            camera_base_path = os.path.join(g_out_fld, rideID[:16])
            for _id, camera in enumerate(edata.cameras):
                offset = tp_reference - camera.start_timestamp
                if start + offset >= 0:
                    if export_videos:

                        # Wait for video cut to end (at most MAX_VIDEO_CUT_THREADS at the same time)
                        start_wait = time.time()
                        while len(video_cut_pocs) > MAX_VIDEO_CUT_THREADS:
                            finished_idxs = []
                            for p_idx, (i_no, p) in enumerate(video_cut_pocs):
                                if p.poll() is not None:
                                    finished_idxs.append(p_idx)

                            for idx in finished_idxs:
                                print(f"Finished: {video_cut_pocs[idx][0]} with return code: "
                                      f"{video_cut_pocs[idx][1].returncode}")

                            # remove finished processes from list
                            for i in sorted(finished_idxs, reverse=True):
                                del video_cut_pocs[i]

                            if len(video_cut_pocs) > MAX_VIDEO_CUT_THREADS:
                                print(f"Waiting to write video segments ... {len(video_cut_pocs)} "
                                      f"( {time.time() - start_wait})", end="\r")

                                time.sleep(WAIT_TIME)

                        video_file_path = f"{camera_base_path}-{_id}.mov"
                        if not os.path.isfile(video_file_path):
                            video_cut_pocs.append((interval_no,
                                                   run_video_cut_cmd(camera.video_path, VIDEO_SIZE,
                                                                    start + offset, end - start + offset,
                                                                    video_file_path)))
                        else:
                            print(f"Already done: {interval_no} - {video_file_path}")

                    new_cfg = namespace_to_dict(adjust_camera_cfg(camera, start + offset,
                                                                  end + offset))

                    # TODO Must calculate real pts of frames -> after conversion :(
                    new_cfg.pop("pts")

                    data["cameras"].append(new_cfg)
                else:
                    print(f"Camera Bad offset {camera.name}")

            data_path = os.path.join(g_out_fld, rideID[:16]+".json")

            if not os.path.isfile(data_path):

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
                with open(data_path, 'w') as outfile:
                    json.dump(data, outfile, indent=2)
            else:
                print(f"Already done: {interval_no} - {data_path}")

        # TODO Convert phone to json

        # TODO convert speed to json

        # TODO convert steer to json


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='experiment_path', help='Path to experiment to export.')
    arg_parser.add_argument(dest='out_fld', help='Path to save.')
    arg_parser.add_argument('--camera-view-size', default=400, type=int, dest="camera_view_size")
    arg_parser.add_argument('--start-tp', default=0, type=float, dest="start_tp")

    arg_parser = arg_parser.parse_args()

    experiment_path = arg_parser.experiment_path
    out_fld = arg_parser.out_fld

    # experiment_path = "/media/nemodrive0/Samsung_T5/nemodrive/18_nov/session_2/1542563017_log"
    # out_fld = "/media/nemodrive0/Samsung_T5/nemodrive/18_nov/segments_session_2"
    export_videos = True

    main(experiment_path, out_fld, export_videos=export_videos)






