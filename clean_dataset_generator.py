import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

UPB_XLIM_EASTING = [423772., 424808.]  # recorded [423792.118788, 424788.577689]
UPB_YLIM_NORTHING = [4920534., 4921623.]  # recorded [4920554.23386, 4921603.97505]


def main():
    folder_path = ""


if __name__ == '__main__':
    main()

exp = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/"
df = pd.read_pickle("{}/phone.log.pkl".format(exp))
steer = pd.read_csv("{}/steer.csv".format(exp))
speed = pd.read_csv("{}/speed.csv".format(exp))

camera_no = 2
with open(exp+"/camera_{}_timestamp".format(camera_no)) as f:
    camera_start_tp = float(f.read())

speed_tp_offset = camera_start_tp - speed.tp.min()
print("Diff tp speed vs camera {}".format(speed_tp_offset))

plt.plot(speed.tp - speed.tp.min() + speed_tp_offset, speed.speed)


speed[speed.tp > 2108 + speed.tp.min() - speed_tp_offset]

start_bad = speed.loc[46279, "tp"]
end_bad = speed.loc[50713, "tp"]


out_fld = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/bad_segment/"
speed[(speed.tp >= start_bad) & (speed.tp <= end_bad)].to_csv(out_fld + "/speed.csv")
steer[(steer.tp >= start_bad) & (steer.tp <= end_bad)].to_csv(out_fld + "/steer.csv")
df[(df.tp >= start_bad) & (df.tp <= end_bad)].to_pickle(out_fld + "/phone.log.pkl")

print()
df.plot("steer")
fig = plt.figure()
plt.plot(df.tp - df.tp.min(), df.magneticHeading)


rawVector = pd.DataFrame.from_dict(list(df["rawVector"].values))
rawVector["tp"] = df["tp"]

fig = plt.figure()
plt.scatter(rawVector.x[1000], rawVector.y)
plt.show()

acc = pd.DataFrame.from_dict(list(df["acceleration"].values))
acc["tp"] = df["tp"]

plt.plot(acc["tp"], acc.x)


# --
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

i_start = 0
no_points = 1000
plt_data = df[i_start:i_start+no_points]
l, = plt.plot(plt_data.easting, plt_data.northing, lw=2, color='red')
plt.axis(UPB_XLIM_EASTING + UPB_YLIM_NORTHING)
# plt.axis([df.easting.min(), df.easting.max(), df.northing.min(), df.northing.max()])

# axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
# axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0, len(df)-no_points, valinit=i_start)
# samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


def update(val):
    # amp = samp.val
    # print (sfreq.val)
    i_start = int(sfreq.val)
    plt_data = df[i_start:i_start + no_points]
    print (plt_data.easting.values)
    l.set_xdata(plt_data.easting)
    l.set_ydata(plt_data.northing)
    fig.canvas.draw_idle()

sfreq.on_changed(update)

plt.show()



from scipy.interpolate import make_interp_spline, BSpline

gps_unique = df.groupby(['loc_tp']).head(1)

fig = plt.figure()
plt.scatter(gps_unique.easting, gps_unique.northing, s=3.5)
plt.show()

x, y = gps_unique.easting.values, gps_unique.northing.values
xx, yy = np.meshgrid(x, y)
scipy.interpolate.RectBivariateSpline(x, y, )
xnew = np.linspace(x.min(),x.max(),len(gps_unique) * 10) #300 represents number of points to make
# between T.min
# and T.max

spl = make_interp_spline(x, y, k=3) #BSpline object
power_smooth = spl(xnew)

plt.scatter(xnew,power_smooth)
plt.show()

import numpy as np
from scipy.interpolate import bisplrep, splev
import matplotlib.pyplot as plt

pts = gps_unique[["easting", "northing"]].values


tck, u = bisplrep(pts.T, u=None, s=0.0)
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck, der=0)

plt.plot(pts[:,0], pts[:,1], 'ro')
plt.plot(x_new, y_new, 'b--')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def range_spline(x, y):
    t = np.arange(x.shape[0], dtype=float)
    t /= t[-1]
    nt = np.linspace(0, 1, 2000)
    new_x = scipy.interpolate.spline(t, x, nt)
    new_y = scipy.interpolate.spline(t, y, nt)
    return (new_x, new_y)

def dist_spline(x, y):
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]
    nt = np.linspace(0, 1, 2000)
    new_x = scipy.interpolate.spline(t, x, nt)
    new_y = scipy.interpolate.spline(t, y, nt)
    return (new_x, new_y)

x = gps_unique.easting.values[-500:]
y = gps_unique.northing.values[-500:]
# x = np.array([ 0,  3,  6,  6,  8,  11, 8, 6, 6])
# y = np.array([ 0,  1,  4,  7,  5,  -7, -10, -10, -5])
#x = np.array([ 0, 2, 4])
#y = np.array([ 0, 2, 0])


(x1, y1) = range_spline(x,y)
# (x2,y2) = dist_spline(x,y)

plt.plot(x,y, 'o')
plt.plot(x1, y1, label='range_spline')
# plt.plot(x2, y2, label='dist_spline')

plt.legend(loc='best')
plt.show()

