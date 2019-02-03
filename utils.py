from argparse import Namespace
import yaml
import numpy as np
import pandas as pd


def get_closest_index(df, timestamp):
    idx = df.index.get_loc(timestamp, method="nearest")
    data_point = df.iloc[idx]
    found_tp = data_point.name
    dif_tp = timestamp - found_tp
    return data_point, idx, dif_tp


def namespace_to_dict(namespace):
    """ Deep (recursive) transform from Namespace to dict """
    dct = dict()
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def read_cfg(config_file):
    """ Parse yaml type config file. """
    with open(config_file) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    cfg = dict_to_namespace(config_data)
    return cfg


def dict_to_namespace(dct):
    """ Deep (recursive) transform from Namespace to dict """
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def get_nonblocking(queue):
    """ Get without blocking from multiprocessing queue"""
    try:
        resp = queue.get(block=False)
    except Exception as e:
        resp = None
    return resp


def get_local_ip():
    """ Try to get local ip used for internet connection """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    found_ip = s.getsockname()[0]
    s.close()

    return found_ip


def merge_intervals(intervals):
    """ Merge intervals from a list of list of [min, max] intervals """
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for t in s:
        if t[0] > s[m][1]:
            m += 1
            s[m] = t
        else:
            s[m] = [s[m][0], max(t[1], s[m][1])]
    return s[:m+1]


def exclude_intervals(intervals, exclude):
    """ Exclude intervals from list of intervals """
    intervals = merge_intervals(intervals)
    exclude = merge_intervals(exclude)
    print(intervals)
    print(exclude)

    for e_start, e_end in exclude:
        valid_intervals = []
        for start, end in intervals:
            if start < e_start < end:
                valid_intervals.append([start, e_start])
                if e_end < end:
                    valid_intervals.append([e_end, end])
            elif start < e_end < end:
                valid_intervals.append([e_end, end])
            elif e_end <= start or e_start >= end:
                valid_intervals.append([start, end])
        intervals = merge_intervals(valid_intervals)

    return intervals


def get_interval_cnt(df, interval, clm="tp", min_hz=5):
    """ Count number of rows for a period of <interval>s from each row """
    next_tp = 0
    interval_cnt = []
    interval_margin = []
    for idx, row in df.iterrows():
        while (df.iloc[next_tp][clm] - row[clm] <= interval) and next_tp < len(df) - 1:
            next_tp += 1
        interval_cnt.append(next_tp - idx)
        interval_margin.append([idx, next_tp])

    interval_cnt = np.array(interval_cnt)
    interval_margin = np.array(interval_margin)
    reject = interval_margin[interval_cnt < min_hz]

    reject = np.array([df.loc[reject[:, 0]][clm].values,
                       df.loc[reject[:, 1]][clm].values]).transpose()
    merged_intervals = merge_intervals(reject.tolist())

    return interval_cnt, interval_margin, merged_intervals


def get_interval_cnt_disjoint(df, interval, clm="tp", min_hz=5):
    """ Count number of rows for a period of <interval>s from each rounded interval """
    interval_cnt = []
    interval_margin = []

    min_tp_s = np.ceil(df[clm].min())
    max_tp_s = np.floor(df[clm].max())
    data = df[clm]-min_tp_s

    intervals = data.groupby(pd.cut(data, np.arange(0, max_tp_s-min_tp_s, interval))).count()
    reject = intervals[intervals < min_hz].index.codes
    reject = np.array([reject+min_tp_s, reject+min_tp_s+interval]).transpose()
    merged_intervals = merge_intervals(reject.tolist())

    return intervals, merged_intervals


def parse_video_time_format(s, fps=30.):
    """ Format MM:SS.f """
    m, sf = s.split(":")
    m = float(m)
    s, f = [float(x) for x in sf.split(".")]

    time_interval = m * 60. + s + 1. /fps * f
    return time_interval

