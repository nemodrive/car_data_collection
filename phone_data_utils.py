import matplotlib
matplotlib.use('TkAgg')  # <-- THIS MAKES IT FAST!

import pandas as pd
import json
import os
from utils import get_interval_cnt_disjoint

import matplotlib.pyplot as plt
import numpy as np

PHONE_LOG_FILE = "phone.log"
PHONE_FILE = "phone.log.pkl"

FREQ_INTERVAL = 1.
MIN_HZ = 5

UPB_XLIM_GPS = [26.0446, 26.0524]
UPB_YLIM_GPS = [44.4346, 44.4404]

UPB_XLIM_EASTING = [423772., 424808.]  # recorded [423792.118788, 424788.577689]
UPB_YLIM_NORTHING = [4920534., 4921623.]  # recorded [4920554.23386, 4921603.97505]

# MAX_SIZE = max(np.diff(UPB_YLIM_EASTING)[0], np.diff(UPB_XLIM_NORTHING)[0])
# UPB_YLIM_EASTING = [UPB_YLIM_EASTING[0], UPB_YLIM_EASTING[0]+MAX_SIZE]
# UPB_XLIM_NORTHING = [UPB_XLIM_NORTHING[0], UPB_XLIM_NORTHING[0]+MAX_SIZE]


def phone_data_to_df(file_path):
    # file_path = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_1/1542549716_log/phone.log"
    # df = pd.read_csv(file_path, sep=";", header=None)
    # df["idx"] = df.index
    #
    # df_data = df[2]
    line_no = -1
    data = []

    msg_tps = []
    msg_client_id = []
    msg = ""
    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_no += 1

            line = line.split(";")
            if len(line) != 3:
                print("Wrong message format (line {}): \n{}".format(line_no, line))
                continue
            line[2] = line[2].strip()  # Might have new line signal at the end

            if line[2].startswith("{\"location\":"):
                if len(msg) > 0:
                    try:
                        d = json.loads(msg)
                        d["tp"] = msg_tps[0]
                        d["msg_tps"] = msg_tps
                        d["msg_client_ids"] = msg_client_id
                        data.append(d)
                    except:
                        print("ERROR for msg parse (line {}): {}".format(line_no, msg))

                # Restart message "concatenation" process
                msg = line[2]
                msg_tps = [float(line[0])]
                msg_client_id = [line[1]]
            else:
                msg += line[2]
                msg_tps.append(float(line[0]))
                msg_client_id.append(line[1])

    df_processed = pd.DataFrame.from_dict(data)

    return df_processed


def validate_data(experiment_path):
    df = pd.read_pickle(os.path.join(experiment_path, PHONE_FILE))

    # count by FREQ_INTERVAL interval
    # phone_log_hz, margin, margin_tp = get_interval(df, FREQ_INTERVAL, min_hz=2)
    phone_log_hz, margin_tp = get_interval_cnt_disjoint(df, FREQ_INTERVAL, min_hz=2)
    phone_log_hz = phone_log_hz.values

    fig = plt.figure()
    plt.plot(phone_log_hz)
    fig.suptitle("Phone log Hz within a {}s interval".format(FREQ_INTERVAL))
    print("nu inteleg de unde")
    print(pd.Series(phone_log_hz, name="phone_log_hz").describe())
    print("\n")
    print("Intervals with Hz < {}".format(MIN_HZ))
    sum_margin = 0
    for min, max in margin_tp:
        print("{} - {} ({}s)".format(min, max, max-min))
        sum_margin += max-min
    print("Total seconds: {}\n".format(sum_margin))
    plt.show()
    plt.pause(0.0000001)  # Note this correction

    gps = df.groupby(['loc_tp']).head(1)

    print("GPS_INFO")
    print("________ No. unique", len(gps["global"].unique()))

    fig = plt.figure()
    plt.scatter(gps["easting"].values, gps["northing"].values, s=3.5, )
    plt.xlim(UPB_XLIM_EASTING)
    plt.ylim(UPB_YLIM_NORTHING)
    plt.axes().set_aspect('equal')
    plt.title("Full GPS")
    plt.show()
    plt.pause(0.0001)
    print("DONE FULL GPS VIEW")

    fig = plt.figure()
    plt.title("True Heading")
    df.trueHeading.plot()
    plt.show()
    plt.pause(0.0001)
    print("True heading plot")


class ScatterLivePlot:
    def __init__(self, df, data_col, tp_window_size=300., tp_col="tp", xlim=None, ylim=None,
                 aspect=None):
        self.df = df = df.sort_values(by=[tp_col])  # Sort by timestamp

        self.data_col = data_col
        self.tp_col = tp_col
        self.data = data = df[data_col]
        self.tp = tp = df[tp_col]
        self.min_tp = tp.min()
        self.plot_tps = tp - tp.min()
        self.tp_window_size = tp_window_size

        self.fig, self.ax = plt.subplots()
        self.min_data, self.max_data = data.min(), data.max()

        self.fig.suptitle(" - ".join(data_col))
        plt.draw()
        plt.pause(0.000001)

        self.xlim = xlim
        self.ylim = ylim
        self.aspect = aspect

        self.start_idx = start_idx = 0
        end_idx = 1
        while tp.iloc[end_idx] - tp.iloc[start_idx] < tp_window_size:
            end_idx += 1

        self.end_idx = end_idx

    def plot(self, plot_tp):
        """ Center tp_window_size to plot_tp"""
        tp_window_size = self.tp_window_size
        plot_tps = self.plot_tps
        tp = self.tp
        data = self.data
        start_idx = self.start_idx
        end_idx = self.end_idx
        ax = self.ax
        min_data, max_data = self.min_data, self.max_data
        min_tp = self.min_tp

        start_tp = max(plot_tp - tp_window_size // 2, min_tp)
        end_tp = start_tp + tp_window_size

        if start_tp < tp.iloc[start_idx]:
            # Move bck to required timestamp
            while start_tp < tp.iloc[start_idx] and start_idx > 0:
                start_idx -= 1

            while tp.iloc[end_idx] > end_tp and end_idx > 0:
                end_idx -= 1
        else:
            max_idx = len(tp) - 1
            # Move fwd to required timestamp
            while start_tp > tp.iloc[start_idx] and start_idx < max_idx:
                start_idx += 1

            while tp.iloc[end_idx] < end_tp and end_idx < max_idx:
                end_idx += 1

        self.start_idx = start_idx
        self.end_idx = end_idx

        plot_t = plot_tps.iloc[start_idx:end_idx].values

        crt_plt_tp = plot_tp - min_tp
        crt_plt_idx = 0
        if crt_plt_idx < len(plot_t):
            while crt_plt_tp > plot_t[crt_plt_idx] and crt_plt_idx < len(plot_t) - 1:
                crt_plt_idx += 1

        plot_d = data.iloc[start_idx:end_idx].values
        if len(plot_d) > 0:
            ax.clear()
            ax.scatter(plot_d[:, 0], plot_d[:, 1], color="blue", s=3.5,)
            ax.scatter(plot_d[crt_plt_idx, 0], plot_d[crt_plt_idx, 1], color="red", s=5.5,)
            if self.xlim is not None:
                self.ax.set_xlim(*self.xlim)
            if self.ylim is not None:
                self.ax.set_ylim(*self.ylim)
            if self.aspect is not None:
                self.ax.set_aspect(self.aspect)

            # ax.plot([plot_tp - min_tp]*2, [min_data, max_data], color="red")


def async_phone_plot(experiment_path, recv_queue, send_queue):
    phone_plot = PhonePlot(experiment_path)
    print("PHONE READY to start live plot...")

    while True:
        msg = recv_queue.get()
        if msg == -1:
            break

        r = phone_plot.plot(msg)

        plt.show()
        plt.pause(0.0000001)  # Note this correction

        send_queue.put(("phone", r))


def get_gps(df):
    import utm
    gps = pd.DataFrame.from_dict(list(df["location"].values))
    gps["tp"] = df["tp"]
    gps["global"] = gps["x"] * gps["y"]

    gps = gps.assign(**{'easting': -1., 'northing': -1., "zone_no": -1., "zone_letter": ""})

    for idx, row in gps.iterrows():
        easting, northing, zone_no, zone_letter = utm.from_latlon(row["y"], row["x"])
        gps.set_value(idx, "longitude", row["x"])
        gps.set_value(idx, "latitude", row["y"])
        gps.set_value(idx, "altitude", row["z"])
        gps.set_value(idx, "easting", easting)
        gps.set_value(idx, "northing", northing)
        gps.set_value(idx, "zone_no", zone_no)
        gps.set_value(idx, "zone_letter", zone_letter)

    return gps


def get_attitude(df):
    df = pd.read_pickle("/media/nemodrive3/Samsung_T5/nemodrive/25_nov/session_2/1543155398_log/phone.log.pkl")
    attitude = pd.DataFrame.from_dict(list(df["attitudeEularAngles"].values))
    attitude = pd.DataFrame.from_dict(list(df["attitude"].values))
    attitude["tp"] = df["tp"]
    phone_start_tp = attitude.tp.min()
    attitude["tp_rel"] = attitude.tp - attitude.tp.min()

    plt.plot(attitude.tp_rel, attitude.x) # - attitude.x.loc[0])
    plt.plot(attitude.tp_rel, attitude.y) # - attitude.y.loc[0])
    plt.plot(attitude.tp_rel, attitude.z) # - attitude.z.loc[0])
    plt.plot(attitude.tp_rel, attitude.w - attitude.w.loc[0])

    acc = pd.DataFrame.from_dict(list(df["acceleration"].values))
    acc["tp"] = df["tp"]
    phone_start_tp = attitude.tp.min()
    acc["tp_rel"] = acc.tp - acc.tp.min()

    plt.plot(acc.tp_rel, acc.x)
    plt.plot(acc.tp_rel, acc.y)
    plt.plot(acc.tp_rel, acc.z)
    plt.plot(acc.tp_rel, acc.w - acc.w.loc[0])

    attitude_start_move_off = 48.5
    attitude_start_move = phone_start_tp + attitude_start_move_off

    df_can_speed = pd.read_csv("/media/nemodrive3/Samsung_T5/nemodrive/25_nov/session_2/1543155398_log/speed.csv")
    can_first_move_tp = df_can_speed[df_can_speed.speed > 0].iloc[0]["tp"]

    print("Diff: {}".format(can_first_move_tp - attitude_start_move))

    start_camera_2 = (phone_start_tp + 62.5) - 51.18  # - camera_2_move

    Diff: 0.6711456775665283
    Diff: 0.60382080078125
    Diff: 0.909038782119751
    Diff: 0.4892098903656006


def gather_all_phone_logs():
    import glob2

    folder = "/media/andrei/Samsung_T51/nemodrive"
    all_phone_logs = glob2.glob(folder + "/**/" + PHONE_LOG_FILE)

    phone_dfs = [phone_data_to_df(x) for x in all_phone_logs]
    gps = [get_gps(x) for x in phone_dfs]
    merged = [pd.merge(p, g, how="outer") for p, g in zip(phone_dfs, gps)]

    # Write pickle data
    for merged_data, phone_log_path in zip(merged, all_phone_logs):
        print (phone_log_path + ".pkl")
        merged_data.to_pickle(phone_log_path + ".pkl")

    # Plot all data
    eastings = np.concatenate([x.easting.values for x in gps])
    northings = np.concatenate([x.northing.values for x in gps])

    fig = plt.figure()
    plt.scatter(eastings, northings, s=3.5)
    plt.title("Full GPS")
    plt.show()

    e_min, e_max = eastings.min(), eastings.max()
    n_min, n_max = northings.min(), northings.max()
    print("UPB_XLIM_EASTING = [{}, {}]  "
          "# recorded [{}, {}]".format(e_min-20, e_max+20, e_min, e_max))
    print("UPB_YLIM_NORTHING = [{}, {}]  "
          "# recorded [{}, {}]".format(n_min-20, n_max+20, n_min, n_max))

# easting = []
# northing = []
# for _f in f:
#     df = pd.read_pickle(_f)
#     gps = get_gps(df)
#     easting.append([gps.easting.min(), gps.easting.max()])
#     northing.append([gps.northing.min(), gps.northing.max()])
#     print ("done")

# from gps_view import rgb_color_range
# color_start = [10, 10, 10]
# color_end = [200, 200, 200]
#
# df = pd.read_pickle("/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log/phone.log.pkl")
# pts = pd.read_csv("/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log"
#                   "/camera_2_pts.log", header=None)
# gps = get_gps(df)
#
# camera_st_tp = 1542537674.27
# target_tp = pts.loc[58941][0] + camera_st_tp
# idx_phone = (df.tp - target_tp).abs().argmin()
# rows_before = 1000
# rows_after = 10000
#
# data = gps.loc[idx_phone-rows_before: idx_phone+rows_after]
#
# colors = rgb_color_range(color_start, color_end, len(data))
# fig = plt.figure()
# plt.scatter(data["northing"], data["easting"], s=3.5, c=np.array(colors) / 255., alpha=1)
#
# # CRT TP
# plt.scatter(gps.loc[idx_phone]["northing"],
#             gps.loc[idx_phone]["easting"], s=30.5, marker='x',c="red", alpha=1)
# plt.xlim(UPB_XLIM_NORTHING)
# plt.ylim(UPB_YLIM_EASTING)
# plt.axes().set_aspect('equal')
#
# plt.title("Full GPS")
# plt.show()


class PhonePlot:
    def __init__(self, experiment_path, tp_window_size=60.):
        plt.ion()  # Note this correction

        self.df = df = pd.read_pickle(os.path.join(experiment_path, PHONE_FILE))

        # Plot GPS
        gps = df[["tp", "easting", "northing", "latitude", "longitude", "global"]]
        self.gps = gps

        self.plotters = []

        plt_steer = ScatterLivePlot(gps, ["easting", "northing"], tp_window_size=tp_window_size,
                                    xlim=UPB_XLIM_EASTING, ylim=UPB_YLIM_NORTHING, aspect="equal")
        self.plotters.append(plt_steer)

    def plot(self, plot_tp):
        for plotter in self.plotters:
            plotter.plot(plot_tp)

    def get_common_tp(self):
        start_tps = []
        for plotter in self.plotters:
            start_tps.append(plotter.tp.min())
        return np.max(start_tps)


if __name__ == "__main__":
    import time

    phone_path = "/media/nemodrive3/Samsung_T5/nemodrive/18_nov/session_0/1542537659_log"
    phone_log_path = os.path.join(phone_path, "phone.log")

    phone_plot = PhonePlot(phone_path)
    start_tp = phone_plot.get_common_tp()
    print(start_tp)

    t = time.time()
    play_speed = 20.

    while True:
        s = time.time()

        crt_tp = start_tp + play_speed * (time.time() - t)

        phone_plot.plot(crt_tp)

        plt.draw()
        plt.pause(0.00000000001)

        time.sleep(1/10.)
