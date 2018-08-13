import os
import time
import obd
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import pyformulas as pf
import numpy as np

from utils import get_nonblocking, get_closest_index


def plog(s):
    print('[OBD] {}'.format(s))


def get_obd(cfg):
    # obd.logger.setLevel(obd.logging.DEBUG)  # enables all debug information

    commands = cfg.obd.commands
    out_dir = cfg.out_dir
    queue = cfg.queue
    receive_queue = cfg.receive_queue

    port = None
    if hasattr(cfg.obd, "port"):
        port = cfg.obd.port
    else:
        serials = obd.scan_serial()
        plog(serials)

        if len(serials) <= 0:
            plog("[ERROR] No serials found!")
            return 0
        else:
            port = serials[0]
            for c in serials:
                if "rfcomm" in c:
                    port = c

    plog("Connecting on port: {}".format(port))

    connection = obd.Async(portstr=port, fast=False)

    fs = []
    for cmd in commands:
        fs.append(open(os.path.join(out_dir, "obd_{}.log".format(cmd)), "w"))
        fs[-1].flush()

    def function_builder(_id):
        def function(r):
            fs[_id].write("{:0.6f}, {}\n".format(time.time(), r.value))
            fs[_id].flush()
        return function

    my_dynamic_functions = {}
    for idx, cmd in enumerate(commands):
        my_dynamic_functions[cmd] = function_builder(idx)

    for idx, cmd in enumerate(commands):
        connection.watch(getattr(obd.commands, cmd), callback=my_dynamic_functions[cmd])

    # Wait for start command:
    plog("Ready")
    receive_queue.put(True)
    resp = queue.get(block=True)
    if resp:
        plog("Start")
    else:
        return 1

    connection.start()

    # Wait for closing command
    while True:
        res = get_nonblocking(queue)
        time.sleep(1)
        if res:
            break

    for f in fs:
        f.close()

    return 0


class OBDLoader:

    def __init__(self, experiment_path, max_tp=1.0, plot=["SPEED"], plot_hist=5.):
        dirname = experiment_path
        self.max_tp = max_tp
        self.plot_hist = plot_hist

        fls = glob.glob(dirname + "/obd*")
        self.data = data = dict()
        for fl in fls:
            type = re.match("obd_(.*).log", os.path.basename(fl)).group(1)
            df = pd.read_csv(fl, header=None)
            df["value"] = df[1].apply(lambda x: None if x is None else x.split()[0])
            df["value"] = df["value"].apply(lambda x: None if x == "None" else float(x))
            df.set_index(0, inplace=True)
            data[type] = df

        plot_func = dict({
            "SPEED": self.plot_speed
        })
        self.plot = [plot_func[x] for x in plot if x in data.keys() and x in plot_func.keys()]

        # self.fig = plt.figure()
        #
        # self.canvas = np.zeros((480, 640))
        # self.screen = pf.screen(self.canvas, 'Sinusoid')

    def get_closest(self, timestamp, show=True):
        r = []
        for fnc in self.plot:
            ans = fnc(timestamp, show=show)
            r.append(ans)
        return r

    def plot_speed(self, timestamp, show):
        dtn = "SPEED"
        df = self.data[dtn]
        plot_hist = self.plot_hist

        data_point, idx, dif_tp = get_closest_index(df, timestamp)
        if abs(dif_tp) > self.max_tp:
            print("[{}] [ERROR] Reached max tp ({})".format(dtn, dif_tp))
            return dif_tp, None

        if show:
            data_point, start_idx, dif_tp = get_closest_index(df, timestamp-plot_hist)
            print ("{}-{}".format(start_idx, idx))
            df.iloc[start_idx:idx+1]["value"].plot()

        return dif_tp, data_point


