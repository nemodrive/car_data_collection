# Andrei, 2018
"""
    Collect data.
"""

from argparse import ArgumentParser
import os
import time
import yaml
from utils import dict_to_namespace
from copy import deepcopy
import torch.multiprocessing as mp
import subprocess
import signal
from subprocess import PIPE
from argparse import Namespace
import shutil

from utils import get_nonblocking, read_cfg
from get_camera import get_camera
from get_can import get_can
from get_obd import get_obd
from get_phone import get_phone

MULTI_LOGS_CMD = 'gnome-terminal -x sh -c "multitail "{}/*.log; bash""'
TAIL_LOG = 'terminator --command="tail {}"'
WAIT_QUIT = 10


def show_logs(cfg):
    import glob

    def plog(s):
        print('[LOGS] {}'.format(s))

    queue = cfg.queue
    out_dir = cfg.out_dir
    show = cfg.view_type
    receive_queue = cfg.receive_queue

    logs = glob.glob(out_dir + "/*log")
    if len(logs) <= 0:
        receive_queue.put(False)
        plog("No logs to view.")
        return 0

    plog(logs)

    # Configure view Mode
    if show == 0:
        cmd = MULTI_LOGS_CMD.format(out_dir)
    elif show == 1:
        cmd = " ".join(["-f " + x for x in logs])
        cmd = TAIL_LOG.format(cmd)

    # Wait for start command:
    receive_queue.put(True)
    resp = queue.get(block=True)
    if resp:
        plog("Start")
    else:
        return 1

    if show == 0:
        pro = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, bufsize=1, shell=True,
                               preexec_fn=os.setsid)
    elif show == 1:
        pro = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, bufsize=1, shell=True,
                               preexec_fn=os.setsid)

    # Wait for closing command
    while True:
        res = get_nonblocking(queue)
        time.sleep(1)
        if res:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            pro.kill()
            pro.terminate()
            break

    return 0


def plog(s):
    print('[COLLECT] {}'.format(s))


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        '-c', '--config-file', default='configs/default.yaml', type=str,  dest='config_file',
        help='Default configuration file'
    )
    arg_parser.add_argument(
        '--out-dir', default='data', type=str, dest="out_dir"
    )
    arg_parser.add_argument(
        '--log-name', default='log', type=str, dest="log_name"
    )
    arg_parser.add_argument(
        '--view', default=False, action="store_true", dest="view"
    )
    arg_parser.add_argument(
        '--view-only', default=False, action="store_true", dest="view_only"
    )
    arg_parser.add_argument(
        '--view-type', default=-1, type=int, dest="view_type"
    )

    arg_parser = arg_parser.parse_args()
    config_file = arg_parser.config_file
    out_dir = arg_parser.out_dir
    log_name = arg_parser.log_name
    view = arg_parser.view
    view_only = arg_parser.view_only
    start_time = time.time()

    # Generate out folder
    if os.path.isdir(out_dir):
        out_dir = os.path.join(out_dir, "{}_{}".format(int(start_time), log_name))
        if os.path.isdir(out_dir):
            plog("Save dir exists: {}".format(out_dir))
            exit(1)
        os.mkdir(out_dir)
    else:
        plog("Out folder does not exist: {}".format(out_dir))

    if view_only:
        view = view_only

    # Read config
    cfg = read_cfg(config_file)
    collect = cfg.collect
    plog("Collect data:")
    plog(collect)
    receive_queue = mp.Queue(maxsize=100)

    all_proc = []
    all_que = []

    default_cfg = Namespace()
    default_cfg.out_dir = out_dir
    default_cfg.view = view
    default_cfg.view_only = view_only
    default_cfg.view_type = cfg.view_type

    def get_default_cfg():
        new_cfg = deepcopy(default_cfg)
        new_cfg.receive_queue = receive_queue
        return new_cfg

    # ==============================================================================================
    # -- Configs & get cameras

    # TODO show & get names of cameras (v4l2-ctl --list-devices)
    if collect.camera:
        cameras_cfg = cfg.camera
        cameras_ind_cfg = []
        camera_out_que = []
        for id in cameras_cfg.ids:
            new_cfg = get_default_cfg()
            new_cfg.__dict__.update(cameras_cfg.default.__dict__)
            new_cfg.id = id

            camera_out_que.append(mp.Queue(maxsize=100))
            new_cfg.queue = camera_out_que[-1]
            cameras_ind_cfg.append(new_cfg)

        # Init cameras
        play_procs, exp_queues_out = [], []
        for i in range(len(cameras_ind_cfg)):
            play_procs.append(mp.Process(target=get_camera, args=(cameras_ind_cfg[i],)))
            play_procs[i].start()
            time.sleep(0.5)

        all_proc += play_procs
        all_que += camera_out_que
    # ==============================================================================================

    # ==============================================================================================
    # -- Get CAN data
    if collect.can:
        new_cfg = get_default_cfg()
        all_que.append(mp.Queue(maxsize=100))
        new_cfg.queue = all_que[-1]
        new_cfg.can = cfg.can

        all_proc.append(mp.Process(target=get_can, args=(new_cfg,)))
        all_proc[-1].start()
    # ==============================================================================================

    # ==============================================================================================
    # -- Get Phone data
    if collect.phone:
        new_cfg = get_default_cfg()
        new_cfg.__dict__.update(cfg.phone.__dict__)
        all_que.append(mp.Queue(maxsize=100))
        new_cfg.queue = all_que[-1]

        all_proc.append(mp.Process(target=get_phone, args=(new_cfg,)))
        all_proc[-1].start()
    # ==============================================================================================

    # ==============================================================================================
    # -- Get OBD data
    if collect.obd:
        new_cfg = get_default_cfg()
        all_que.append(mp.Queue(maxsize=100))
        new_cfg.obd = cfg.obd
        new_cfg.queue = all_que[-1]

        all_proc.append(mp.Process(target=get_obd, args=(new_cfg,)))
        all_proc[-1].start()

    # ==============================================================================================

    # Wait for everything to start
    time.sleep(10)

    # ==============================================================================================
    # -- Visualize logs
    if cfg.view_type >= 0:
        new_cfg = get_default_cfg()
        all_que.append(mp.Queue(maxsize=100))
        new_cfg.queue = all_que[-1]

        all_proc.append(mp.Process(target=show_logs, args=(new_cfg,)))
        all_proc[-1].start()
    # ==============================================================================================

    # Receive ready signals:
    ready_signals = 0
    while ready_signals < len(all_proc):
        receive_queue.get(block=True)
        ready_signals += 1
        plog("Ready signals: {}".format(ready_signals))

    # Send start signals
    plog("Send ok commands")
    for que in all_que:
        que.put(True)

    while True:
        q = raw_input("[COLLECT] Wait command: [q] - send quit to children | [Q] - quit app\n")

        # Send quit to all threads
        if q == "q":
            plog("Sending quit")
            for que in all_que:
                que.put(True)
            plog("Wait to send quit command...")
            time.sleep(10)

        # Quit app and terminate processes
        elif q == "Q":
            plog("Exit")
            break

        print("")

    # QUIT
    # Close processes
    for i in range(len(all_proc)):
        all_proc[i].terminate()
        all_proc[i].join()

    if view_only:
        shutil.rmtree(out_dir)

