import pandas as pd
from utils import read_cfg
import os
from argparse import Namespace
import copy

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
        print(d.cameras[-1].name)

        # TODO Add camera start move frame

    return d
