import cv2
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT
import time

from utils import get_nonblocking

FFMPEG_COMMAND = "ffmpeg -f v4l2 -framerate {}" \
                 " -video_size {}x{} " \
                 "-input_format mjpeg -i " \
                 "/dev/video{} -c copy {}.mkv"
TRY_OPEN = 5


def get_camera(cfg):

    cam_id = cfg.id
    view = cfg.view
    view_only = cfg.view_only
    res = tuple(cfg.res)
    fps = cfg.fps
    out_path = cfg.out_dir
    view_height = cfg.view_height
    queue = cfg.que
    receive_queue = cfg.receive_queue

    out = None

    prefix_log = "[Camera {}] ".format(str(cam_id))

    def plog(s):
        print(prefix_log + s)

    # Wait for start command:
    plog("Ready")
    receive_queue.put(True)
    resp = queue.get(block=True)
    if resp:
        plog("Start")
    else:
        return 1

    # Record using ffmpeg (Recommended)
    if out_path and not view:
        plog("Record with ffmpeg.")
        path = os.path.join(out_path, 'f_camera_{}'.format(cam_id))
        log_path = os.path.join(out_path, 'f_camera_{}.log'.format(cam_id))
        cmd = FFMPEG_COMMAND.format(fps, res[0], res[1], cam_id, path)
        with open(log_path, "w") as outfile:
            pro = subprocess.Popen(cmd,
                                   stdout=PIPE, stderr=outfile, stdin=PIPE, bufsize=1,
                                   shell=True)

        # Wait for closing command
        while True:
            res = get_nonblocking(queue)
            time.sleep(1)
            if res:
                pro.communicate(input="q\n".encode('utf-8'))
                pro.stdin.close()
                break

        return 0

    # Start camera with opencv
    plog("Open camera.")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        plog("Error opening resource.")
        return 1

    # Record using opencv
    if out_path and not view_only:
        plog("Record with opencv.")

        codec = cfg.cv2_codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        path = os.path.join(out_path, 'c_camera_{}.avi'.format(cam_id))
        out = cv2.VideoWriter(path, fourcc, fps, res)

    # opencv Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    for i in range(TRY_OPEN):
        rval, frame = cap.read()

    if view:
        scale_view = view_height / float(frame.shape[0])

    # custom adjust
    scale_view = 3
    w = 1280
    h = 720
    #

    key = -1
    while rval:
        if view:
            frame_view = cv2.resize(frame, (0, 0), fx=scale_view, fy=scale_view)

            #
            frame_view = frame_view[h:h*2, w:2*w]
            h_, w_ = frame_view.shape[:2]
            frame_view[:, int(w_/2)-2:int(w_/2)+2] = 255
            frame_view[int(h_/2)-2:int(h_/2)+2, :] = 255
            #

            cv2.imshow("Stream: {}".format(str(cam_id)), frame_view)
            key = cv2.waitKey(1)

        rval, frame = cap.read()

        if out:
            out.write(frame)

        res = get_nonblocking(queue)

        if key == 27 or key == 1048603 or res:
            break

    cap.release()
    if out:
        out.release()

    # TODO compress ffmpeg video
    return 0
