# Andrei, 2018
"""
    Collect data.
"""
from argparse import ArgumentParser
import numpy as np
import cv2
import os

from utils import read_cfg
from get_camera import VideoLoad
from get_obd import OBDLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='experiment_path', help='Path to experiment to visualize.')
    arg_parser.add_argument('--camera-view-size', default=400, type=int, dest="camera_view_size")

    arg_parser = arg_parser.parse_args()

    experiment_path = arg_parser.experiment_path
    camera_view_size = arg_parser.camera_view_size

    cfg = read_cfg(os.path.join(experiment_path, "cfg.yaml"))
    cfg_extra = read_cfg(os.path.join(experiment_path, "cfg_extra.yaml"))

    collect = cfg.collect
    record_timestamp = cfg.record_timestamp

    video_loders = []
    obd_loader = None

    if collect.camera:
        camera_names = ["camera_{}".format(x) for x in cfg.camera.ids]
        camera_cfgs = [getattr(cfg.camera, x) for x in camera_names]
        extra_camera_cfgs = [getattr(cfg_extra, x) for x in camera_names]
        video_loders = [VideoLoad(experiment_path, x, getattr(cfg_extra, x),
                                  view_height=camera_view_size,
                                  flip_view=getattr(cfg.camera, x).flip) for x in camera_names]

    # if collect.obd:
    #     obd_loader = OBDLoader(experiment_path)
    #
    cursor_img = np.zeros((100, 100, 3)).astype(np.uint8)

    def get_key():
        cv2.imshow("Cursor", cursor_img)
        k = cv2.waitKey(0)
        r = chr(k % 256)
        return r

    freq_tp = [1/30., 1/10., 1.]
    freq_id = 0
    freq = freq_tp[freq_id]
    r = None
    crt_tp = record_timestamp

    # plt.ion()
    # plt.show()

    while r != "q":
        r = get_key()
        plt.clf()

        if r == ".":
            # Add fps
            crt_tp += freq
        elif r == ",":
            # Prev frame
            crt_tp -= freq
        elif r == "f":
            freq_id = (freq_id + 1) % len(freq_tp)
            freq = freq_tp[freq_id]
            print("Speed: {}".format(freq))

        if collect.camera:
            print ("------")
            for v in video_loders:
                dif_tp, frame = v.get_closest(crt_tp)
                # print (dif_tp)
                v.show(frame)
        if collect.obd:
            obd_data = obd_loader.get_closest(crt_tp)

        # plt.show()
        # plt.pause(0.0001)  # Note this correction
