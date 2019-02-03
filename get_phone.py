from websocket_server import WebsocketServer
import time
import os
import torch.multiprocessing as mp
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
import pyformulas as pf
import numpy as np
import json

from utils import get_nonblocking, get_local_ip


def plog(s):
    print('[Phone] {}'.format(s))


def new_client(client, server):
    """Called for every client connecting (after handshake)"""
    plog("New client connected with id {} and address {}, {}".format(client['id'],
                                                                     client['address'][0],
                                                                     client['address'][1]))


def client_diconnected(client, server):
    """Called for every client disconnecting"""
    plog("Time {:.6f}".format(time.time()))
    plog("Client(%d) disconnected" % client['id'])


def std_message_received(client, server, message):
    """Called when a client sends a message"""
    # plog("Client(%d) said: %s" % (client['id'], message))


def run_server_classic(ip_address, port):
    """run_server"""
    server = WebsocketServer(port, host=ip_address)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_diconnected)
    server.set_fn_message_received(std_message_received)
    server.run_forever()


def run_server(i_ip_address, i_port, i_message_received, i_new_client, i_client_diconnected):
    """run_server"""
    server = WebsocketServer(i_port, host=i_ip_address)
    server.set_fn_new_client(i_new_client)
    server.set_fn_client_left(i_client_diconnected)
    server.set_fn_message_received(i_message_received)
    server.run_forever()


def get_phone(cfg):
    ip = cfg.ip
    port = cfg.port
    out_dir = cfg.out_dir
    queue = cfg.queue
    receive_queue = cfg.receive_queue

    log = False
    global phone_received_msgs
    global phone_write_to_file
    phone_write_to_file = torch.zeros(1)
    phone_received_msgs = torch.zeros(1)
    phone_write_to_file.share_memory_()
    phone_received_msgs.share_memory_()

    if ip == 0:
        ip = get_local_ip()
        plog("Found ip: {}".format(ip))
    log_path = "{}/phone.log".format(out_dir)
    out_file = open(os.path.join(log_path), "w")

    plog("Star server on: {}:{}".format(ip, port))

    # Configure message received
    def message_received(client, server, message):
        """Called when a client sends a message"""
        global phone_received_msgs
        global phone_write_to_file
        phone_received_msgs += 1
        if log:
            plog("Client(%d) said: %s" % (client['id'], message))
        if phone_write_to_file[0] > 0:
            out_file.write("{:0.6f};{};{}\n".format(time.time(), client['id'], message))

    # Start server:
    proc = mp.Process(target=run_server, args=(ip, port, message_received, new_client,
                                               client_diconnected))
    proc.start()

    while phone_received_msgs[0] <= 0:
        time.sleep(1)

    # Wait for start command:
    plog("Ready")
    receive_queue.put(True)
    resp = queue.get(block=True)
    if resp:
        plog("Start")
    else:
        return 1

    phone_write_to_file[0] = 1

    # Wait for closing command
    while True:
        res = get_nonblocking(queue)
        time.sleep(1)
        if res:
            break

    proc.terminate()
    proc.join()

    return 0







# class PHONELoader:
#
#     def __init__(self, experiment_path, max_tp=1.0, plot=["SPEED"], plot_hist=5.):
#         dirname = experiment_path
#         self.max_tp = max_tp
#         self.plot_hist = plot_hist
#
#         fls = glob.glob(dirname + "/obd*")
#         self.data = data = dict()
#         for fl in fls:
#             type = re.match("obd_(.*).log", os.path.basename(fl)).group(1)
#             df = pd.read_csv(fl, header=None)
#             df["value"] = df[1].apply(lambda x: None if x is None else x.split()[0])
#             df["value"] = df["value"].apply(lambda x: None if x == "None" else float(x))
#             df.set_index(0, inplace=True)
#             data[type] = df
#
#         plot_func = dict({
#             "SPEED": self.plot_speed
#         })
#         self.plot = [plot_func[x] for x in plot if x in data.keys() and x in plot_func.keys()]
#
#         # self.fig = plt.figure()
#         #
#         # self.canvas = np.zeros((480, 640))
#         # self.screen = pf.screen(self.canvas, 'Sinusoid')
#
#     def get_closest(self, timestamp, show=True):
#         r = []
#         for fnc in self.plot:
#             ans = fnc(timestamp, show=show)
#             r.append(ans)
#         return r
#
#     def plot_speed(self, timestamp, show):
#         dtn = "SPEED"
#         df = self.data[dtn]
#         plot_hist = self.plot_hist
#
#         data_point, idx, dif_tp = get_closest_index(df, timestamp)
#         if abs(dif_tp) > self.max_tp:
#             print("[{}] [ERROR] Reached max tp ({})".format(dtn, dif_tp))
#             return dif_tp, None
#
#         if show:
#             data_point, start_idx, dif_tp = get_closest_index(df, timestamp-plot_hist)
#             print ("{}-{}".format(start_idx, idx))
#             df.iloc[start_idx:idx+1]["value"].plot()
#
#         return dif_tp, data_point
#
