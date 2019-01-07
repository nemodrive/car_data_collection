import pandas as pd
import json
import os
from utils import get_interval_cnt_disjoint
import matplotlib.pyplot as plt
import numpy as np

# PHONE_FILE = "phone.log"
PHONE_FILE = "phone.log.pkl"

FREQ_INTERVAL = 1.
MIN_HZ = 5

UPB_XLIM_GPS = [26.0446, 26.0524]
UPB_YLIM_GPS = [44.4346, 44.4404]

UPB_YLIM_EASTING = [443390.0, 444383.]  # recorded [443410.34934956284, 444363.32093989343]
UPB_XLIM_NORTHING = [2880479., 2881903.]  # recorded [2880499.7528450321, 2881883.3800112251]

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
    print(pd.Series(phone_log_hz, name="phone_log_hz").describe())
    print("\n")
    print("Intervals with Hz < {}".format(MIN_HZ))
    sum_margin = 0
    for min, max in margin_tp:
        print("{} - {} ({}s)".format(min, max, max-min))
        sum_margin += max-min
    print("Total seconds: {}\n".format(sum_margin))


class ScatterLivePlot:
    def __init__(self, df, data_col, tp_window_size=300., tp_col="tp", xlim=None, ylim=None,
                 aspect=None):
        self.df = df = df.sort(tp_col)  # Sort by timestamp

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

        start_tp = max(plot_tp - tp_window_size//2, min_tp)
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
        while crt_plt_tp > plot_t[crt_plt_idx]:
            crt_plt_idx += 1

        plot_d = data.iloc[start_idx:end_idx].values
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


def get_gps(df):
    import utm
    gps = pd.DataFrame.from_dict(list(df["location"].values))
    gps["tp"] = df["tp"]
    gps["global"] = gps["x"] * gps["y"]

    gps = gps.assign(**{'easting': -1., 'northing': -1., "zone_no": -1., "zone_letter": ""})

    for idx, row in gps.iterrows():
        easting, northing, zone_no, zone_letter = utm.from_latlon(row["x"], row["y"])
        gps.set_value(idx, "easting", easting)
        gps.set_value(idx, "northing", northing)
        gps.set_value(idx, "zone_no", zone_no)
        gps.set_value(idx, "zone_letter", zone_letter)

    return gps

easting = []
northing = []
for _f in f:
    df = pd.read_pickle(_f)
    gps = get_gps(df)
    easting.append([gps.easting.min(), gps.easting.max()])
    northing.append([gps.northing.min(), gps.northing.max()])
    print ("done")


class PhonePlot:
    def __init__(self, experiment_path, tp_window_size=60.):
        plt.ion()  # Note this correction

        self.df = df = pd.read_pickle(os.path.join(experiment_path, PHONE_FILE))

        # Plot GPS
        gps = get_gps(df)
        self.gps = gps

        print("GPS_INFO")
        print("________ All:", len(gps))
        print("________ unique", len(gps["global"].unique()))

        # fig = plt.figure()
        # plt.scatter(gps["x"].values, gps["y"].values)
        fig = plt.figure()
        plt.scatter(gps["northing"].values, gps["easting"].values, s=3.5,)
        plt.xlim(UPB_XLIM_NORTHING)
        plt.ylim(UPB_YLIM_EASTING)
        plt.axes().set_aspect('equal')

        plt.title("Full GPS")
        plt.show()
        plt.pause(0.0001)
        key = raw_input("Press key to continue ...")

        self.plotters = []

        plt_steer = ScatterLivePlot(gps, ["northing", "easting"], tp_window_size=tp_window_size,
                                    xlim=UPB_XLIM_NORTHING, ylim=UPB_YLIM_EASTING, aspect="equal")
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

    phone_path = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_1/1542549716_log"
    phone_log_path = os.path.join(phone_path, "phone.log")
    #
    # df = phone_data_to_df(phone_log_path)
    # df.to_pickle(phone_log_path + ".pkl")
    #
    phone_plot = PhonePlot(phone_path)
    start_tp = phone_plot.get_common_tp()
    print(start_tp)

    t = time.time()
    play_speed = 20.

    while True:
        s = time.time()

        crt_tp = start_tp + play_speed * (time.time() - t)

        phone_plot.plot(crt_tp)

        plt.plot()

        plt.show()
        plt.pause(0.00000000001)

        time.sleep(1/10.)
