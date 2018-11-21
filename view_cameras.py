# Andrei, 2018
"""
    Interactive script to view multiple cameras.

    sudo python -m pip install --upgrade pip setuptools wheel
    sudo /usr/local/bin/pip install opencv-python

"""

from argparse import ArgumentParser
import cv2
import numpy as np
import os
import time

CROSS_COLOR = np.array([0, 0, 255], dtype=np.uint8)
FONT_SCALE = 1.5
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (0, 255, 0)

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument('cameras', help='Camera ids', type=int, nargs="+")
    arg_parser.add_argument('--scale', default=1., type=float, help='View resize factor.')
    arg_parser.add_argument('--fps', default=30, type=int, help='What FPS to read from video at.')
    arg_parser.add_argument('--res', default=[1920, 1080], type=int,
                            nargs=2, help='Video resolution.')
    arg_parser.add_argument('--cross-size', default=2, type=int, help='Cross pixel size.')
    arg_parser.add_argument('--save', default="data/", help='Save folder.')

    args = arg_parser.parse_args()
    cameras = args.cameras
    scale = args.scale
    fps = args.fps
    res = args.res
    cross_size = args.cross_size
    save_folder = args.save

    caps = dict({})

    for camera in cameras:
        cap = cv2.VideoCapture("/dev/video{}".format(camera))
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera {} not working!".format(camera))
            continue

        print("Camera {} working!".format(camera))

        # Configure camera
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # Set camera buffer size to 0
        caps[camera] = cap

    save_path = os.path.join(save_folder, "{}_camera_{}_off_{}ms.jpg")

    save_screen_shot = False
    show_cross = True
    show_menu = True

    def screen_shot():
        global save_screen_shot
        save_screen_shot = True

    def toggle_cross():
        global show_cross
        show_cross = not show_cross

    def toggle_menu():
        global show_menu
        show_menu = not show_menu

    menu = dict({
        27: (quit, "Key [ESC]: Exit"),  # if the 'ESC' key is pressed, Quit
        ord('s'): (screen_shot, "Key [s]: Save screen shots"),
        ord('c'): (toggle_cross, "Key [c]: Toggle cross"),
        ord('m'): (toggle_menu, "Key [m]: Toggle menu"),
    })

    menu_text = "\n".join([x[1] for x in menu.values()])

    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the 'ESC' key is pressed, Quit
        if key in menu.keys():
            menu[key][0]()
        elif key != 255:
            print("Unknown key: {}".format(key))

        tp_common = time.time()

        for camera, cap in caps.items():
            ret, frame = cap.read()

            view_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = view_frame.shape[:2]

            if show_cross:
                view_frame[h//2-cross_size: h//2+cross_size, :] = CROSS_COLOR
                view_frame[:, w//2-cross_size: w//2+cross_size] = CROSS_COLOR

            if save_screen_shot:
                view_frame[:, :, 2] += 100
                off = int((time.time() - tp_common) * 1000)
                cv2.imwrite(save_path.format(tp_common, camera, off), frame)

            if show_menu:
                y0, dy = 50, 50
                for i, line in enumerate(menu_text.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(view_frame, line, (50, y), FONT, FONT_SCALE, FONT_COLOR)

            cv2.imshow("Camera: {}".format(camera), view_frame)

        # Reset save screen shot
        save_screen_shot = False
