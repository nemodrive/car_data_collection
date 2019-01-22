import cv2
import random
import numpy as np


with open("/home/andrei/Documents/images.txt", "r") as f:
    data = f.readlines()

video = "/media/andrei/CE04D7C504D7AF291/nemodrive/data_collect/data/1533228223_log/camera_2.mkv"
# Normal
vid = cv2.VideoCapture(video)
# vid.get(cv2.CAP_PROP_FRAME_COUNT)

dup = 0
prev_frame = None
for i in range(n-1):
    ret, frame = vid.read()
    # if np.all(frame == prev_frame):
    #     dup += 1
    # prev_frame = frame

frames_true = [frame]
for i in range(2):
    ret, frame = vid.read()
    frames_true.append(frame)

# go to
for i in range(-100, 100):
    vid = cv2.VideoCapture(video)
    vid.set(cv2.CAP_PROP_POS_FRAMES, n+i)
    ret, frame = vid.read()
    print (i)
    for i_frame in frames_true:
        if np.all(i_frame == frame):
            print "WOW"


# Dist check
max_n = 2987

#get
vid = cv2.VideoCapture(video)
all_frames = []
frame_tp = []
for i in range(max_n):
    frame_tp.append(vid.get(cv2.CAP_PROP_POS_MSEC))
    ret, frame = vid.read()
    all_frames.append(frame)

#compare
diff = []
for i in range(max_n):
    vid = cv2.VideoCapture(video)
    vid.set(cv2.CAP_PROP_POS_MSEC, frame_tp[i])
    ret, frame = vid.read()
    found = False

    for offset in range(100):
        if offset > 0:
            for sign in range(2):
                offset = -1 * offset
                go = i + offset
                if 0 <= go <= len(frame_tp):
                    if np.all(frame == all_frames[go]):
                        diff.append(frame_tp[go] - frame_tp[i])
                        found = True
                        break
        else:
            if np.all(frame == all_frames[i]):
                diff.append(frame_tp[i] - frame_tp[i])
                found = True
                break

    if not found:
        diff.append(np.inf)

cv2.imshow("test", frames_true[0]); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imshow("test", frame); cv2.waitKey(0); cv2.destroyAllWindows()
