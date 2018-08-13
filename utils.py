from argparse import Namespace
import yaml


def get_closest_index(df, timestamp):
    idx = df.index.get_loc(timestamp, method="nearest")
    data_point = df.iloc[idx]
    found_tp = data_point.name
    dif_tp = timestamp - found_tp
    return data_point, idx, dif_tp


def namespace_to_dict(namespace):
    """Deep (recursive) transform from Namespace to dict"""
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
    """Deep (recursive) transform from Namespace to dict"""
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
    """Try to get local ip used for internet connection"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    found_ip = s.getsockname()[0]
    s.close()

    return found_ip
