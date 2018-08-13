"""
Usefull links: http://bengreen.eu/fancyhtml/quickreference/ffmpegv4l2.html

v4l2-ctl --list-devices

v4l2-ctl --device /dev/video0 --all
ffmpeg -f v4l2 -list_formats all -i /dev/video1
v4l2-ctl --list-formats

v4l2-ctl -d /dev/video1 --list-ctrls
v4l2-ctl -d /dev/video1 -c <option>=<value>
ffmpeg -i video.mkv -c copy video_fixed.mkv
ffprobe -v error -show_entries frame=pkt_pts_time -of default=noprint_wrappers=1:nokey=1 camera_0.mkv > camer_0_pts.log

"""

import cv2
import os
import subprocess
from subprocess import PIPE
import time
import pandas as pd
import glob
import numpy as np
import math

from utils import get_nonblocking

# FFMPEG_COMMAND = "ffmpeg -f v4l2 -framerate {}" \
#                  " -video_size {}x{} " \
#                  "-input_format mjpeg -i " \
#                  "/dev/video{} -c copy {}.mkv"
FFMPEG_COMMAND = "ffmpeg -f v4l2" \
                 " -video_size {}x{} " \
                 "-i /dev/video{} -c copy {}.mkv"
TRY_OPEN = 5

CAMERA_MATRIX = [
    np.array([[734.30239003, 0., 640.],
              [0., 734.30239003, 360.],
              [0., 0., 1.]]),
    np.array([[1166.5659572746586, 0., 640.],
              [0., 869.11688245431424, 360.],
              [0., 0., 1.]]),
]

RVEC = np.array([0, 0, 0], np.float)  # rotation vector
TVEC = np.array([0, 0, 0], np.float)  # translation vector


def get_camera(cfg):

    cam_id = cfg.id
    view = cfg.view
    view_only = cfg.view_only
    res = tuple(cfg.res)
    fps = cfg.fps
    out_path = cfg.out_dir
    view_height = cfg.view_height
    queue = cfg.queue
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
        path = os.path.join(out_path, 'camera_{}'.format(cam_id))
        log_path = os.path.join(out_path, 'camera_{}.log'.format(cam_id))
        # cmd = FFMPEG_COMMAND.format(fps, res[0], res[1], cam_id, path)
        cmd = FFMPEG_COMMAND.format(res[0], res[1], cam_id, path)
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
    calibrate = False
    if calibrate:
        scale_view = 1
        w = 1280
        h = 720
        #
    flip = [0, 0, 0, 0, 0]
    key = -1
    while rval:
        if view:
            if flip[cam_id]:
                frame = cv2.flip(frame, -1)

            frame_view = cv2.resize(frame, (0, 0), fx=scale_view, fy=scale_view)

            #
            if calibrate:
                # frame_view = frame_view[h:h*2, w:2*w]
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


class VideoLoad:
    def __init__(self, experiment_path, camera_name, cfg_camera,
                 frame_buffer_size=500, view_height=400, flip_view=0, max_tp=1.0):
        dirname = experiment_path
        filename = camera_name
        fls = glob.glob(os.path.join(dirname, filename) + ".*")
        fls = [x for x in fls if not x.endswith("log")]
        assert len(fls) == 1, "Cannot find {} ({})".format(filename, fls)

        self.cfg_camera = cfg_camera
        print(cfg_camera)
        self.max_tp = max_tp
        self.filename = filename
        self.video_file_path = fls[0]
        self.vid = cv2.VideoCapture(fls[0])
        self.frame_buffer_size = frame_buffer_size
        self.pts_file = pd.read_csv(os.path.join(dirname, "{}_pts.log".format(filename)),
                                    header=None)

        with open(os.path.join(dirname, "{}_timestamp".format(filename))) as infile:
            lines = infile.readlines()
            start_timestamp = float(lines[0])

        self.pts_file["timestamp"] = self.pts_file[0].apply(lambda x: x + start_timestamp)
        self.pts_file.set_index("timestamp", inplace=True)

        self.scale_view = view_height / self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.flip_view = flip_view

        self.buffer = dict()
        self.vid_next_frame = 0
        self.min_buffer = 0
        self.ret = True

        self.last_emit = -1

    def next_frame(self):
        frame_data = self.get_frame(self.last_emit+1)
        if frame_data is not None:
            self.last_emit = frame_data["frame_no"]
        return frame_data

    def prev_frame(self, ):
        frame_data = self.get_frame(self.last_emit-1)
        if frame_data is not None:
            self.last_emit = frame_data["frame_no"]
        return frame_data

    def reset_vid(self):
        self.vid = cv2.VideoCapture(self.video_file_path)
        self.buffer = dict()
        self.vid_next_frame = 0
        self.min_buffer = 0
        self.ret = True

    def get_frame(self, frame_no):
        if frame_no in self.buffer:
            return self.buffer[frame_no]
        elif frame_no >= self.vid_next_frame:
            ret = self.ret
            vid = self.vid
            buffer = self.buffer
            buffer_size = self.frame_buffer_size
            vid_next_frame = self.vid_next_frame
            pts_file = self.pts_file
            min_buffer = self.min_buffer

            b = None
            while vid_next_frame <= frame_no and ret:
                ret, frame = vid.read()
                if ret:
                    b = dict({"frame": frame,
                              "frame_no": vid_next_frame,
                              "pts": pts_file.iloc[vid_next_frame]})
                    buffer[vid_next_frame] = b
                    if len(buffer) > buffer_size:
                        buffer.pop(min_buffer)
                        min_buffer += 1

                    vid_next_frame += 1
            self.ret = ret
            self.vid_next_frame = vid_next_frame
            self.min_buffer = min_buffer
            return b
        elif 0 <= frame_no < self.vid_next_frame:
            self.reset_vid()
            return self.get_frame(0)

        return None

    def get_closest(self, timestamp):
        df = self.pts_file
        idx = df.index.get_loc(timestamp, method="nearest")
        data_point = df.iloc[idx]
        found_tp = data_point.name
        dif_tp = timestamp - found_tp
        if abs(dif_tp) > self.max_tp:
            print("[{}] [ERROR] Reached max tp ({})".format(self.filename, dif_tp))
            return dif_tp, None

        data_point = self.get_frame(idx)
        return dif_tp, data_point

    def show(self, data, playback=False):
        if not isinstance(data, dict):
            print("[{}] [ERROR] No frame ({})".format(self.filename, data))
            return False

        if "frame" not in data:
            print("[{}] [ERROR] No frame ({})".format(self.filename, data))
            return False

        frame = data["frame"]
        scale_view = self.scale_view
        flip = self.flip_view

        if flip:
            frame = cv2.flip(frame, -1)

        cfg = self.cfg_camera
        # Draw lines
        camera_matrix = CAMERA_MATRIX[cfg.matrix]
        rvec = np.array([cfg.rvec])
        tvec = np.array([cfg.tvec])
        y_pos = cfg.y_pos
        wheel_w = cfg.wheel_w
        car_offset_x = cfg.car_offset_x
        max_z = cfg.max_z

        pts3d = np.array([
            [wheel_w, y_pos, 0.0],
            [wheel_w, y_pos, max_z],
            [-wheel_w, y_pos, 0.0],
            [-wheel_w, y_pos, max_z]
        ])
        pts3d[:, 0] = pts3d[:, 0] + car_offset_x
        y, x = frame.shape[0] /2., frame.shape[1] /2.

        imagePoints, jacobian = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, None)
        frame[int(y - 1): int(y + 1)] = 0
        frame[:, int(x - 1): int(x + 1)] = 0

        for i in range(1, imagePoints.shape[0], 2):
            frame = cv2.line(frame,
                             tuple(imagePoints[i - 1, 0].astype(np.int)),
                             tuple(imagePoints[i, 0].astype(np.int)),
                             (0, 0, 255), thickness=1)

        frame_view = cv2.resize(frame, (0, 0), fx=scale_view, fy=scale_view)

        cv2.imshow(self.filename, frame_view)
        return True




if __name__ == "__main__":
    import numpy as np

    v = VideoLoad("data/1533228223_log/camera_0.mkv")

    frame = v.next_frame()
    r = None

    cursor_img = np.zeros((100, 100, 3)).astype(np.uint8)

    def get_key():
        cv2.imshow("Cursor", cursor_img)
        k = cv2.waitKey(0)
        r = chr(k % 256)
        return r

    freq_tp = [1, 10, 30]
    freq_id = 0
    freq = freq_tp[freq_id]
    while r != "q":
        r = get_key()

        if r == ",":
            # Next frame
            for _ in range(freq):
                frame = v.prev_frame()
            v.show(frame)
        elif r == ".":
            # Prev frame
            for _ in range(freq):
                frame = v.next_frame()
            v.show(frame)
        elif r == "f":
            freq_id = (freq_id + 1) % len(freq_tp)
            freq = freq_tp[freq_id]
            print("Speed: {}".format(freq))


# vid = cv2.VideoCapture("data/1533228223_log/camera_0.mkv")
# select_rand = [123, 3000, 6341, 7123, 9341, 8412, 5131, 11123, 12341]
# bck = dict()
# ret = True
# frame_no = 0
# while ret:
#     ret, frame = vid.read()
#     frame_no += 1
#     if frame_no in select_rand:
#         bck[frame_no] = frame
#

if __name__ == "__main__":
    import cv2
    import numpy as np
    vid = cv2.VideoCapture("data/1533228223_log/camera_0.mkv")
    ret = True
    frame_no = 0
    while ret:
        ret, frame = vid.read()

    frame_cpy = frame.copy()

    resolution = [1280., 720.]
    a_w, a_h = 16., 9.  # Aspect ratio
    w, h = resolution
    d_fov = 120.
    h_fov = np.rad2deg(2. * np.arctan(np.tan(np.deg2rad(d_fov)/2.) * np.cos(np.arctan(a_h / a_w))))
    v_fov = np.rad2deg(2. * np.arctan(np.tan(np.deg2rad(d_fov)/2.) * np.sin(np.arctan(a_h / a_w))))

    # x and y are the X and Y dimensions of the image produced
    #   by the camera, measured from the center of the image
    x = w/2.
    y = h/2.

    # axis skew
    s = 0.

    # focal lengths of the camera in the X and Y directions
    # given the FOV a_x in the horizontal direction
    f_x = x / np.tan(np.deg2rad(h_fov) / 2.)
    f_y = y / np.tan(np.deg2rad(v_fov) / 2.)

    f_x = x / np.tan(np.deg2rad(57.5) / 2.)
    f_y = y / np.tan(np.deg2rad(45) / 2.)


    camera_matrix = np.array(
        [
            [f_x, s, x],
            [0, f_y, y],
            [0, 0, 1.]
        ]
    )

    rvec = np.array([0,0,0], np.float) # rotation vector
    tvec = np.array([0,0,0], np.float) # translation vector


    def points_in_circum(r, n=1000, get=1):
        n = int(n * r)
        l = [(math.cos(2 * np.pi / n * x) * r, math.sin(2 * np.pi / n * x) * r) for x in
             xrange(0, int((n + 1)/ 4.))]
        l = np.array(l)

        g = (l[:, 1] < get).sum()
        print(g)
        l = l[:g]
        return l


    y_pos = 0.26
    wheel_w = 0.19
    car_offset_x = 0.
    max_z = 10.
    # circ = points_in_circum(100)
    # pts3d = np.concatenate([circ[:, 0].reshape((-1, 1)),
    #                         np.zeros((circ.shape[0], 1)),
    #                         circ[:, 1].reshape((-1, 1))], axis=1)
    #
    # pts3d[:, 0] = pts3d[:, 0] * wheel_w
    # pts3d[:, 1] = y_pos
    # pts3d[:, 2] = pts3d[:, 2] * max_z
    # print (pts3d[0])
    # print (pts3d[-1])
    pts3d = np.array([
        [wheel_w, y_pos, 0.0],
        [wheel_w, y_pos, 10.0],
        [-wheel_w, y_pos, 0.0],
        [-wheel_w, y_pos, 10.0]
    ])
    pts3d[:, 0] = pts3d[:, 0] + car_offset_x

    frame = frame_cpy.copy()
    imagePoints, jacobian = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, None)
    frame[int(y-1): int(y+1)] = 0
    frame[:, int(x-1): int(x+1)] = 0

    for i in range(1, imagePoints.shape[0], 2):
        frame = cv2.line(frame,
                         tuple(imagePoints[i-1, 0].astype(np.int)),
                         tuple(imagePoints[i, 0].astype(np.int)),
                         (0, 0, 255), thickness=1)
    # frame = cv2.line(frame,
    #                  tuple(imagePoints[2, 0].astype(np.int)), tuple(imagePoints[3, 0].astype(
    #         np.int)),
    #                  (0, 0, 255), thickness=2)

    cv2.imshow("Test", frame); cv2.waitKey(0); cv2.destroyAllWindows()
