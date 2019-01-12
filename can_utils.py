import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')  # <-- THIS MAKES IT FAST!

import matplotlib.pyplot as plt
import numpy as np
from utils import get_interval_cnt_disjoint

SPEED_FILE = "speed.csv"
STEER_FILE = "steer.csv"

FREQ_INTERVAL = 1.
MIN_HZ = 5

CMD_NAMES = dict({
    # cmd_name, data_name, can_id
    "speed": ("SPEED_SENSOR", "SPEED_KPS", "354"),
    "steer": ("STEERING_SENSORS", "STEER_ANGLE"),
    "brake": ("BRAKE_SENSOR", "PRESIUNE_C_P")
})

DBC_FILE = "logan.dbc"


def validate_data(experiment_path):
    plt.ion()
    speed = pd.read_csv(os.path.join(experiment_path, SPEED_FILE))
    steer = pd.read_csv(os.path.join(experiment_path, STEER_FILE))

    # Speed Data
    print("="*30, " Speed info ", "="*30)
    fig = plt.figure()
    speed.speed.plot(title="Speed")
    print(speed.speed.describe())
    print("\n")

    # count by FREQ_INTERVAL interval
    # speed_log_hz, margin, margin_tp = get_interval_cnt(speed, FREQ_INTERVAL, min_hz=MIN_HZ)
    speed_log_hz, margin_tp = get_interval_cnt_disjoint(speed, FREQ_INTERVAL, min_hz=MIN_HZ)
    speed_log_hz = speed_log_hz.values

    fig = plt.figure()
    plt.plot(speed_log_hz)
    fig.suptitle("Speed log Hz within a {}s interval".format(FREQ_INTERVAL))
    print(pd.Series(speed_log_hz, name="speed_log_hz").describe())
    print("\n")
    print("Intervals with Hz < {}".format(MIN_HZ))
    for min, max in margin_tp:
        print("{} - {} ({}s)".format(min, max, max-min))
    print("\n")

    print("="*30, " Steering info ", "="*30)
    fig = plt.figure()
    steer.steer.plot(title="steer")
    print(steer.steer.describe())
    print("\n")

    steer_log_hz, margin_tp = get_interval_cnt_disjoint(steer, FREQ_INTERVAL, min_hz=MIN_HZ)
    steer_log_hz = steer_log_hz.values

    fig = plt.figure()
    plt.plot(steer_log_hz)
    fig.suptitle("Steer log Hz within a {}s interval".format(FREQ_INTERVAL))
    print(pd.Series(steer_log_hz, name="steer_log_hz").describe())
    print("\n")

    print("Intervals with Hz < {}".format(MIN_HZ))
    for min, max in margin_tp:
        print("{} - {} ({}s)".format(min, max, max-min))

    print("="*70)
    print("")
    plt.show()
    plt.pause(0.0000001)  # Note this correction


class DataframeLivePlot:
    def __init__(self, df, data_col, tp_window_size=60., tp_col="tp"):
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
        self.fig.suptitle(data_col)

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

        plot_t = plot_tps.iloc[start_idx:end_idx]
        ax.clear()
        ax.plot(plot_t.values, data.iloc[start_idx:end_idx].values, color="blue")
        ax.plot([plot_tp - min_tp]*2, [min_data, max_data], color="red")
        # self.fig.savefig("test.svg")
        # self.fig.canvas.draw()


class CanPlot:
    def __init__(self, experiment_path, tp_window_size=60.):
        plt.ion() ## Note this correction

        self.plotters = []

        self.speed = pd.read_csv(os.path.join(experiment_path, SPEED_FILE))
        plt_speed = DataframeLivePlot(self.speed, "speed", tp_window_size=tp_window_size)
        self.plotters.append(plt_speed)

        self.steer = pd.read_csv(os.path.join(experiment_path, STEER_FILE))
        plt_steer = DataframeLivePlot(self.steer, "steer", tp_window_size=tp_window_size)
        self.plotters.append(plt_steer)

    def plot(self, plot_tp):
        for plotter in self.plotters:
            plotter.plot(plot_tp)
        return True

    def get_common_tp(self):
        start_tps = []
        for plotter in self.plotters:
            start_tps.append(plotter.tp.min())
        return np.max(start_tps)


def async_can_plot(experiment_path, recv_queue, send_queue):
    can_plot = CanPlot(experiment_path)

    while True:
        msg = recv_queue.get()

        if msg == -1:
            break

        r = can_plot.plot(msg)

        plt.show()
        plt.pause(0.0000001)  # Note this correction

        send_queue.put(("can", r))


if __name__ == "__main__":
    import time
    experiment_path = "/media/andrei/Samsung_T51/nemodrive/18_nov/session_0/1542537659_log"
    # validate_data(experiment_path)

    can_plot = CanPlot(experiment_path)
    start_tp = can_plot.get_common_tp()
    print(start_tp)

    t = time.time()
    play_speed = 20.

    first = False
    while True:
        s = time.time()

        crt_tp = start_tp + play_speed * (time.time() - t)

        can_plot.plot(crt_tp)

        plt.draw()
        plt.pause(0.00000000001)

        time.sleep(1/30.)
        print(time.time()-s)
