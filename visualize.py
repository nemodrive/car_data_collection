# Andrei, 2018
"""
    Collect data.
"""
from argparse import ArgumentParser
import numpy as np
import cv2
import os
import time

from utils import read_cfg
from get_camera import VideoLoad
from get_obd import OBDLoader
import matplotlib.pyplot as plt

from can_utils import validate_data as validate_can_data
from can_utils import CanPlot

from phone_data_utils import validate_data as validate_phone_data
from phone_data_utils import PhonePlot

CAN_PLOT_TIME = 5000

CFG_FILE = "cfg.yaml"
CFG_EXTRA_FILE = "cfg_extra.yaml"

PLAYBACK_FACTOR = 2
CV_WAIT_KEY_TIME = 1

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='experiment_path', help='Path to experiment to visualize.')
    arg_parser.add_argument('--camera-view-size', default=400, type=int, dest="camera_view_size")

    arg_parser = arg_parser.parse_args()

    experiment_path = arg_parser.experiment_path
    camera_view_size = arg_parser.camera_view_size

    cfg = read_cfg(os.path.join(experiment_path, CFG_FILE))
    cfg_extra = read_cfg(os.path.join(experiment_path, CFG_EXTRA_FILE))

    collect = cfg.collect
    record_timestamp = cfg.recorded_min_max_tp
    common_min_max_tp = cfg.common_min_max_tp

    video_loders = []
    obd_loader = None
    can_plot = None
    phone_plot = None
    plot_stuff = False
    live_plot = True

    if collect.camera:
        camera_names = ["camera_{}".format(x) for x in cfg.camera.ids]
        camera_cfgs = [getattr(cfg.camera, x) for x in camera_names]
        extra_camera_cfgs = [getattr(cfg_extra, x) for x in camera_names]
        video_loders = [VideoLoad(experiment_path, x, getattr(cfg_extra, x),
                                  view_height=camera_view_size,
                                  flip_view=getattr(cfg.camera, x).flip) for x in camera_names]

    if collect.obd:
        obd_loader = OBDLoader(experiment_path)

    if plot_stuff:
        plt.ion()
        # plt.show()

    print("=" * 70)
    if collect.can:
        print("=" * 30, "Validate can data", "=" * 30)
        validate_can_data(experiment_path)
        key = raw_input("Press key to continue ...")
        print("")

    print("=" * 70)
    if collect.phone:
        print("=" * 30, "Validate phone data", "=" * 30)
        validate_phone_data(experiment_path)
        key = raw_input("Press key to continue ...")
        print("")

    if live_plot:
        print("=" * 70)
        if collect.can:
            plot_stuff = True
            can_plot = CanPlot(experiment_path)
            key = raw_input("Press key to continue ...")
            print("")

        print("=" * 70)
        if collect.phone:
            plot_stuff = True
            phone_plot = PhonePlot(experiment_path)
            key = raw_input("Press key to continue ...")
            print("")

    print("=" * 70)
    cursor_img = np.zeros((100, 100, 3)).astype(np.uint8)

    def get_key(wait_time):
        cv2.imshow("Cursor", cursor_img)
        k = cv2.waitKey(wait_time)
        # r = chr(k % 256)
        r = k & 0xFF
        return r

    freq_tp = [1/30., 1/10., 1.]
    freq_id = 0
    freq = freq_tp[freq_id]
    r = None
    crt_tp = common_min_max_tp[0]
    print ("START factor: --->")
    print (crt_tp)
    print ("------------------")

    live_play = False
    key_wait_time = 0
    playback_speed = 1.
    playback_factor = PLAYBACK_FACTOR


    # -- Define menu

    # Menu functions
    def increase_tp():
        global crt_tp
        global freq
        crt_tp += freq

    def decrease_tp():
        global crt_tp
        global freq
        crt_tp -= freq

    def change_freq():
        global freq
        global freq_id
        freq_id = (freq_id + 1) % len(freq_tp)
        freq = freq_tp[freq_id]

    def toggle_play():
        global live_play
        global key_wait_time
        live_play = not live_play
        key_wait_time = CV_WAIT_KEY_TIME if live_play else 0

    def increase_playback_speed():
        global playback_speed
        playback_speed = playback_speed * playback_factor

    def decrease_playback_speed():
        global playback_speed
        playback_speed = playback_speed / playback_factor

    menu = dict({
        27: (quit, "Key [ESC]: Exit"),  # if the 'ESC' key is pressed, Quit
        ord('l'): (change_freq, "Key [ l ]: Change freq"),
        ord('\''): (increase_tp, "Key [ \'; ]: Increase tp by freq"),
        ord(';'): (decrease_tp, "Key [ ; ]: Decrease tp by freq"),

        ord('p'): (toggle_play, "Key [ p ]: Toggle Live play"),
        ord(']'): (increase_playback_speed, "Key [ ] ]: Increase playback speed (*{})"),
        ord('['): (decrease_playback_speed, "Key [ [ ]: Decrease playback speed"),
    })

    menu_text = "\n".join([x[1] for x in menu.values()])

    while r != "q":
        prev_tp = time.time()
        key = get_key(key_wait_time)
        plt.clf()

        if key in menu.keys():
            menu[key][0]()
        elif key != 255:
            print("Unknown key: {}".format(key))

        if collect.camera:
            print ("------")
            frames = []
            for v in video_loders:
                dif_tp, frame = v.get_closest(crt_tp)
                frames.append(frame)
                # print (dif_tp)
                v.show(frame)

        if collect.obd:
            obd_data = obd_loader.get_closest(crt_tp)

        if collect.can:
            can_plot.plot(crt_tp)

        if collect.phone:
            phone_plot.plot(crt_tp)

        # TODO Plot magnetometer

        if plot_stuff:
            plt.show()
            plt.pause(0.0000001)  # Note this correction

        if live_play:
            crt_tp += (time.time() - prev_tp) * playback_factor
