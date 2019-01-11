import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


exp = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/bad_segment/"
df = pd.read_pickle("{}/phone.log.pkl".format(exp))
steer = pd.read_csv("{}/steer.csv".format(exp))
speed = pd.read_csv("{}/speed.csv".format(exp))

speed["mps"] = speed.speed * 1000 / 3600.

plt.scatter(steer.tp, steer.steering, s=3.5)
plt.scatter(speed.tp, speed.speed, s=3.5)

dt = 1.

def get_next_pos(dt, speed, )
dist = crt_speed * dt
fc = 2 * np.pi * r
p = distance / fc
step = 2 * np.pi * p / float(no_points)
points = np.array([
    (math.cos(step * x) * r, math.sin(step * x) * r) for x in range(0, no_points + 1)
])
if center_x:
    points[:, 0] = points[:, 0] - r
