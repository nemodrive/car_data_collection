# Andrei, 2018
"""
    Collect data.
"""
import matplotlib
matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!

from argparse import ArgumentParser
import numpy as np
import cv2
import os
import time

from utils import read_cfg
from get_camera import async_camera_draw
import matplotlib.pyplot as plt

from can_utils import validate_data as validate_can_data
from can_utils import CanPlot

from phone_data_utils import validate_data as validate_phone_data
from phone_data_utils import PhonePlot
from multiprocessing import Process, Queue

plt.ion()  ## Note this correction

CAN_PLOT_TIME = 5000
FONT_SCALE = 1.0
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (0, 255, 0)

CFG_FILE = "cfg.yaml"
CFG_EXTRA_FILE = "cfg_extra.yaml"

PLAYBACK_FACTOR = 2
CV_WAIT_KEY_TIME = 1

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='experiment_path', help='Path to experiment to visualize.')
    arg_parser.add_argument('--camera-view-size', default=400, type=int, dest="camera_view_size")
    arg_parser.add_argument('--start-tp', default=0, type=float, dest="start_tp")

    arg_parser = arg_parser.parse_args()

    experiment_path = arg_parser.experiment_path
    camera_view_size = arg_parser.camera_view_size
    start_tp = arg_parser.start_tp

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

    processes = []
    recv_queue = Queue()
    recv_processes = []
    send_queues = []

    ref_tp = 0  # Main camera reference timestamp
    if collect.camera:
        camera_names = ["camera_{}".format(x) for x in cfg.camera.ids]
        camera_cfgs = [getattr(cfg.camera, x) for x in camera_names]
        extra_camera_cfgs = [getattr(cfg_extra, x) for x in camera_names]

        with open(f"{experiment_path}/{camera_names[0]}_timestamp") as infile:
            lines = infile.readlines()
            ref_tp = float(lines[0])

        # Async mode
        video_loders = []
        for camera_name in camera_names:

            cfg_camera = getattr(cfg_extra, camera_name)
            camera_view_size = camera_view_size
            flip_view = getattr(cfg.camera, camera_name).flip

            video_plot = Queue()

            p = Process(target=async_camera_draw,
                        args=(experiment_path, camera_name, cfg_camera, camera_view_size,
                              flip_view, video_plot, recv_queue))
            processes.append(p)
            video_loders.append(video_plot)
            send_queues.append(video_plot)
            recv_processes.append("camera_name")

        # video_loders = [VideoLoad(experiment_path, x, getattr(cfg_extra, x),
        #                           view_height=camera_view_size,
        #                           flip_view=getattr(cfg.camera, x).flip) for x in camera_names]
        #

    # Not working in python 2.7
    # if collect.obd:
    #     obd_loader = OBDLoader(experiment_path)

    plt.ion()
    if plot_stuff:
        plt.show()

    print("=" * 70)
    if collect.can:
        print("=" * 30, "Validate can data", "=" * 30)
        validate_can_data(experiment_path)
        key = input("Press key to continue ...")
        print("")

    print("=" * 70)
    if collect.phone:
        print("=" * 30, "Validate phone data", "=" * 30)
        validate_phone_data(experiment_path)
        key = input("Press key to continue ...")
        print("")

    if live_plot:
        if collect.can:
            print("=" * 70, "Can")
            plot_stuff = True
            can_plot = CanPlot(experiment_path)

            # # ASYNC MODE [ Not working because of matplotlib ]
            # can_plot = Queue()
            #
            # p = Process(target=async_can_plot,
            #             args=(experiment_path, can_plot, recv_queue))
            # processes.append(p)
            # send_queues.append(can_plot)
            # recv_processes.append("can")

            # key = input("Press key to continue ...")
            print("")

        if collect.phone:
            print("=" * 70, "Phone")
            plot_stuff = True
            phone_plot = PhonePlot(experiment_path)

            # # ASYNC MODE [ Not working because of matplotlib ]
            # phone_plot = Queue()
            #
            # p = Process(target=async_phone_plot,
            #             args=(experiment_path, phone_plot, recv_queue))
            # processes.append(p)
            # send_queues.append(phone_plot)
            # recv_processes.append("phone")

            # key = input("Press key to continue ...")
            print("")

    for p in processes:
        p.start()

    print("=" * 70, "Start")

    freq_tp = [1/30., 1/10., 1/5., 1/3., 1/2., 1., 2., 5., 10.]
    freq_id = 0
    freq = freq_tp[freq_id]
    r = None
    crt_tp = common_min_max_tp[0]  #+ 60.
    if start_tp != 0:
        crt_tp = start_tp

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
        print("New frequency: {}".format(freq))

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

    # SHOW menu

    show_menu = True
    cursor_img = np.zeros((300, 300, 3)).astype(np.uint8)


    def get_key(wait_time, tp=0):
        if show_menu:
            y0, dy = 10, 25
            cursor_img = np.zeros((300, 300, 3)).astype(np.uint8)
            new_text = menu_text + f"\nTP:{tp}"
            for i, line in enumerate(new_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(cursor_img, line, (50, y), FONT, FONT_SCALE, FONT_COLOR)

        cv2.imshow("Cursor", cursor_img)
        k = cv2.waitKey(wait_time)
        # r = chr(k % 256)
        r = k & 0xFF
        return r

    while r != "q":
        prev_tp = time.time()
        key = get_key(key_wait_time, tp=crt_tp)
        # plt.clf()

        if key == 27:
            break
        if key in menu.keys():
            menu[key][0]()
        elif key != 255:
            print("Unknown key: {}".format(key))

        # if collect.obd:
        #     obd_data = obd_loader.get_closest(crt_tp)
        if collect.can:
            # can_plot.put(crt_tp)
            can_r = can_plot.plot(crt_tp)
            # crt_steer, crt_speed = (np.random.rand() - 0.5) * 500, None
            crt_steer, crt_speed = None, None

            if "steer" in can_r:
                crt_steer = can_r["steer"]["steer"]
            if "speed" in can_r:
                crt_speed = can_r["speed"]["mps"]

        if collect.camera:
            for v in video_loders:
                v.put([0, crt_tp, crt_steer])
                # dif_tp, frame = v.get_closest(crt_tp)
                # v.show(frame)

        if collect.phone:
            # phone_plot.put(crt_tp)
            phone_plot.plot(crt_tp)

        # # send bulk messages to Queus
        # for send_queue in send_queues:
        #     send_queue.put(crt_tp)

        # TODO Plot magnetometer

        if plot_stuff:
            plt.show()
            plt.pause(0.0000001)  # Note this correction

        recv_sync = 0
        responses = []
        while len(responses) < len(recv_processes):
            r = recv_queue.get()
            responses.append(r)

        if live_play:
            crt_tp += (time.time() - prev_tp) * playback_factor

        print(f"crt_tp: {crt_tp} ({crt_tp - ref_tp}) \t Steer: {crt_steer} \t speed: {crt_speed}")

    for q in send_queues:
        q.put(-1)

    for p in processes:
        p.join()
