from argparse import Namespace


def namespace_to_dict(namespace):
    """Deep (recursive) transform from Namespace to dict"""
    dct = dict()
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


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
