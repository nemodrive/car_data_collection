import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# df = pd.read_csv("/media/andrei/Seagate Expansion Drive/nemodrive/1539434900_log/phone.log", sep=";", header=None)
#
# # Transform old df
# x = df[1]
# df[1] = x.apply(lambda x: int(x[:1]))
# df[2] = x.apply(lambda x: x[2:])
# df.to_csv("/media/andrei/CE04D7C504D7AF291/nemodrive/data_collect/data/1533228223_log/phone.log",
#           sep=";", float_format='%.6f', index=False, header=False)

def rgb_to_no(rgb):
    return (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]


def no_to_rgb(no):
    return [no >> 16, (no % 65536) >> 8, no % 256]


def rgb_color_range(start, end, cnt):
    no_start = rgb_to_no(start)
    no_end = rgb_to_no(end)

    if no_end - no_start <= cnt:
        return None

    colors = []
    for no in np.linspace(no_start, no_end, cnt).round():
        no = int(no)
        clr = no_to_rgb(no)
        colors.append(clr)
    return colors


def phone_data_to_df(file_path):
    df = pd.read_csv(file_path, sep=";", header=None)
    df["idx"] = df.index

    df_data = df[2]
    data = []
    for idx in df_data.index:
        try:
            d = json.loads(df_data.loc[idx])
            d["idx"] = idx
            data.append(d)
        except:
            continue

    df_phone = pd.DataFrame.from_dict(data)
    df_processed = pd.merge(df, df_phone, on="idx")

    return df_processed


file_dir = "/media/andrei/Samsung_T51/nemodrive/15_nov/1542296320_log/"
file_path = file_dir + "phone.log"

df_phone = phone_data_to_df(df[2])
print (len(df_phone))

camera_start_tp = 1539434914.730855
gps = pd.DataFrame.from_dict(list(df["location"].values))
gps["tp"] = df[0]
gps["global"] = gps["x"] * gps["y"] * gps["x"]
print("GPS_INFO")
print ("all", len(gps))
print ("unique", len(gps["global"].unique()))
pause = False

# ==================================================================================================
# -- GPS view

gps = pd.DataFrame.from_dict(list(df["location"].values))
gps["tp"] = df["tp"]
gps["global"] = gps["x"] * gps["y"]
print("GPS_INFO")
print ("all", len(gps))
print ("unique", len(gps["global"].unique()))
pause = False

plt.scatter(gps["x"].values, gps["y"].values)

# ==================================================================================================

color_start = [10, 10, 10]
color_end = [200, 200, 200]

def onclick(event):
    global pause
    pause = not pause


fig = plt.figure()
dpi = 100
fig.set_size_inches(18.5, 10.5)

fig.canvas.mpl_connect('button_press_event', onclick)
fig_save_path = file_dir + "gps_segments/plot_{}.png"
clrs_save_path = file_dir + "gps_segments/plot_{}.csv"
i_start = 0
idx = 0
dif = 0.0001
ax_dim = [gps["x"].min() - dif, gps["x"].max() + dif,
          gps["y"].min() - dif, gps["y"].max() + dif,]


f = 0.0001
while True:
    if not pause:
        print("Figure: ", idx)
        i_end = IMPORTANT_POS[idx]
        idx += 1
        fig.clear()
        start_tp = camera_start_tp + i_start
        end_tp = camera_start_tp + i_end
        filter_gps = gps[(gps["tp"] > start_tp) & (gps["tp"] < end_tp)].drop_duplicates("global")
        colors = rgb_color_range(color_start, color_end, len(filter_gps))
        plt.scatter(filter_gps["x"], filter_gps["y"], s=3.5, c=np.array(colors)/255., alpha=1)

        # -- Interpolate between gps points
        prev_tp = 0
        for index, row in filter_gps.iterrows():
            tp = row["tp"]
            if prev_tp > 0:
                filter_phone = df[(df[0] >= prev_tp) & (df[0] < tp)]
                x = np.linspace(prev_row["x"], row["x"], len(filter_phone))
                y = np.linspace(prev_row["y"], row["y"], len(filter_phone))
                plt.scatter(x, y, s=16.4)
                p_i = 0
                for p_idx, p_row in filter_phone.iterrows():
                    a = np.deg2rad((360.-p_row["trueHeading"]+90) % 360)
                    plt.plot([x[p_i], x[p_i]+np.cos(a)*f],
                             [y[p_i], y[p_i]+np.sin(a)*f])
                    p_i += 1
            prev_tp = tp
            prev_row = row

        # radius = 0.5;
        # CenterX = 0.5;
        # CenterY = 0.5;
        # angle = np.deg2rad(0-90)
        #
        # x = np.cos(angle) * radius + CenterX;
        # Y = np.sin(angle) * radius + CenterY;
        # print(x, Y)
        #
        # --

        i_start = i_end
        plt.axis(ax_dim)
        plt.draw()
        pd.Series(colors).to_csv(clrs_save_path.format(idx-1))
        fig.savefig(fig_save_path.format(idx-1), dpi=dpi, transparent=True)
        pause = True
        print ("unique", len(filter_gps["global"].unique()))

    fig.canvas.get_tk_widget().update() # process events


# Decode img
file_path = "/media/andrei/Seagate Expansion Drive/nemodrive/1539434900_log/gps_segments/plot_2.png"
img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
img = cv2.cv
img.resize(img.shape[0] * img.shape[1], 4)
df_pxpd = pd.DataFrame(img)
df_pxpd["bgr"] = img[:, :3].tolist()
df_pxpd["no"] = df_pxpd["bgr"].apply(rgb_to_no)
unique_clrs = df_pxpd["no"].unique()

df = pd.read_csv("/media/andrei/Seagate Expansion "
                 "Drive/nemodrive/1539434900_log/gps_segments/plot_2.csv", header=None)

orig = df["no"].unique()


unique_clrs.sort()
dif_clrs = unique_clrs[1:] - unique_clrs[:-1]
dif_clrs = pd.Series(dif_clrs)
print(len(df_pxpd["no"].unique()))


import time
import numpy as np

rows,cols = img.shape
t = []


for i in range(90):
    st = time.time()
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    t.append(time.time()-st)
np.mean(t)


