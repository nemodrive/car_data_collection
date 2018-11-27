# Irina, 2018

import time
import serial
import signal
from argparse import ArgumentParser
import subprocess
from subprocess import Popen, PIPE, STDOUT
import cv2
import numpy as np
import os

CANDUMP_CMD = "candump -L {} > {}"
WIN_SIZE = 512
FONT_SCALE = 1.5
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (0, 255, 0)
DEFAULT_DELAY = 200

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument('serial', help='Serial id', type=int)
    arg_parser.add_argument('--no-can', default=False, action="store_true", dest="no_can")
    arg_parser.add_argument('--can', default='can0', type=str, dest="can")
    arg_parser.add_argument('--save-dir', default='data/', type=str, dest="save_dir")
    arg_parser = arg_parser.parse_args()

    serial_id = arg_parser.serial
    save_can = not arg_parser.no_can
    save_dir = arg_parser.save_dir

    can = arg_parser.can

    usb_serial = "/dev/ttyUSB{}".format(serial_id)
    ser = serial.Serial(usb_serial, 115200)
    # ser.timeout = 1
    #
    # # not sure if should be set or not
    # # ser.xonxoff = False     # disable software flow control
    # # ser.rtscts = False     # disable hardware (RTS/CTS) flow control
    # # ser.dsrdtr = False       # disable hardware (DSR/DTR) flow control
    # # ser.writeTimeout = 2     # timeout for write
    #

    step_size = 1  # left step
    delay = 300  # delay should be at least around 200-300

    # ==============================================================================================
    # Menu description
    MENU = "Change step:        i | u\n" \
           "Control:            arrow keys\n" \
           "Delay step:         k | l\n" \
           "Enable | disable:   e | q"
    view_frame = np.zeros((WIN_SIZE, WIN_SIZE, 3), dtype=np.uint8)

    y0, dy = 50, 50
    for i, line in enumerate(MENU.split('\n')):
        y = y0 + i * dy
        cv2.putText(view_frame, line, (50, y), FONT, FONT_SCALE, FONT_COLOR)

    cv2.imshow("Test", view_frame)
    # ==============================================================================================

    file_tp = time.time()

    if save_can:
        save_dir = os.path.join(save_dir, "control_{}".format(int(file_tp)))
        os.mkdir(save_dir)
        can_file = "{}/commands_{}.can".format(save_dir, file_tp)
        can_cmd = CANDUMP_CMD.format(can_file, can)

        pro = subprocess.Popen(can_file, stdout=PIPE, stderr=PIPE, stdin=PIPE, bufsize=1,
                               shell=True, preexec_fn=os.setsid)
        time.sleep(0.1)
        poll = pro.poll()
        assert poll is None, "Could not save candump!"

    f = open("{}/commands_{}.txt".format(save_dir, file_tp), "w")


    def send_cmd(cmd_msg):
        # ser.write(cmd_msg + "\n")
        f.write("{},{}\n".format(cmd_msg, str(time.time())))


    # Send default delay
    cmd = "d:{}".format(DEFAULT_DELAY)
    send_cmd(cmd)
    print("Set default delay: {}".format(cmd))

    running = True
    while running:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # if the 'ESC' key is pressed, Quit
            running = False
            continue
        elif key == ord('i'):  # increase step
            step_size = step_size + 1
            msg = "step is {}".format(step_size)
            cmd = None
        elif key == ord('u'):  # decrease step
            step_size = step_size - 1
            msg = "step is {}".format(step_size)
            cmd = None
        elif key == ord('k'):  # increase delay
            delay += 100
            cmd = "d:" + repr(delay)
            msg = "delay is {}".format(delay)
        elif key == ord('l'):  # decrease delay
            delay = max(200, delay - 100)
            cmd = "d:" + repr(delay)
            msg = "delay is {}".format(delay)
        elif key == 81:
            cmd = "l:" + repr(step_size)
            msg = "go left by {}".format(step_size)
        elif key == 83:
            cmd = "r:" + repr(step_size)
            msg = "go right by {}".format(step_size)
        elif key == ord('e'):
            cmd = "e:1"
            msg = "Enable"
        elif key == ord('q'):
            cmd = "e:1"
            msg = "Disable"
        elif key == 255:
            continue
        else:
            print("Unknown key {}".format(key))
            continue

        print(msg)

        if cmd is not None:
            send_cmd(cmd)

    ser.close()

    if save_can:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
