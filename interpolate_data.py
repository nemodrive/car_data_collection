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

x = approx_path_mean.easting.values[:1000]
y = approx_path_mean.northing.values[:1000]
# x = np.array([ 0,  3,  6,  6,  8,  11, 8, 6, 6])
# y = np.array([ 0,  1,  4,  7,  5,  -7, -10, -10, -5])
#x = np.array([ 0, 2, 4])
#y = np.array([ 0, 2, 0])


(x1, y1) = range_spline(x,y)
# (x2,y2) = dist_spline(x,y)

fig = plt.figure()
plt.plot(x,y, 'o')
plt.plot(x1, y1, label='range_spline')
# plt.plot(x2, y2, label='dist_spline')

plt.legend(loc='best')
plt.show()
