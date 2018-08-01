import os
import time
import obd

from utils import get_nonblocking


def plog(s):
    print('[OBD] {}'.format(s))


def get_obd(cfg):
    obd.logger.setLevel(obd.logging.DEBUG)  # enables all debug information

    commands = cfg.obd.commands
    out_dir = cfg.out_dir
    queue = cfg.queue
    receive_queue = cfg.receive_queue

    plog(obd.scan_serial())

    if len(commands) <= 0:
        return 0

    connection = obd.Async()

    fs = []
    for cmd in commands:
        fs.append(open(os.path.join(out_dir, "{}.log".format(cmd)), "w"))

    # a callback to save to file
    def save_cmd(id, r):
        fs[id].write("{:0.6f}, {}\n".format(time.time(), r.value))

    # Wait for start command:
    plog("Ready")
    receive_queue.put(True)
    resp = queue.get(block=True)
    if resp:
        plog("Start")
    else:
        return 1

    for idx, cmd in enumerate(commands):
        connection.watch(getattr(obd.commands, "RPM"), callback=lambda x: save_cmd(idx, x))
        connection.start()

    # Wait for closing command
    while True:
        res = get_nonblocking(queue)
        time.sleep(1)
        if res:
            break

    return 0
