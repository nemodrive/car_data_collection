from car_utils import get_rotation, get_car_can_path, get_points_rotated
from street_view import ImageWgsHandler
import sys
import time
import threading
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

loopBool = True
orientation, offset_x, offset_y = 0., 0., 0.
fig_gps, fig_path = None, None
ax_gps, ax_path = None, None
base_coord = None
gps_split = None
loaded = False
interval_idx = -1


def closeLooping():
    global loopBool
    loopBool = False


def looping():
    global loopBool, loaded
    global orientation, offset_x, offset_y
    global fig_gps, fig_path
    global ax_gps, ax_path
    global base_coord
    global gps_split
    global interval_idx

    g_out_fld = export_info["g_out_fld"]
    for idx, (tp_start, tp_end) in enumerate(cut_intervals):
        print(f"{idx} - {tp_start} : {tp_end}")
        interval_idx = idx

        tp_start += tp_reference
        tp_end += tp_reference

        rideID = export_info["rideIDs"][idx]
        camera_base_path = os.path.join(g_out_fld, rideID[:16])
        camera_path = f"{camera_base_path}-{0}"
        if os.path.isfile(camera_path):
            vid = cv2.VideoCapture(camera_path)
            first_frame, ret = vid.read()
            vid.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            last_frame, ret = vid.read()

            cv2.imshow("First frame", first_frame)
            cv2.imshow("Last frame", last_frame)
            cv2.waitKey(1)

        phone_split = phone[(phone.tp >= tp_start) & (phone.tp < tp_end)]
        steer_split = steer[(steer.tp >= tp_start) & (steer.tp < tp_end)]
        speed_split = speed[(speed.tp >= tp_start) & (speed.tp < tp_end)]
        gps_split = gps_unique[(gps_unique.tp >= tp_start) & (gps_unique.tp < tp_end)]

        can_coord = get_car_can_path(speed_split.copy(), steer_split.copy())

        guess_orientation = np.random.uniform(0, 360)
        guess_offest_x = np.random.uniform(-4, 4)
        guess_offest_y = np.random.uniform(-4, 4)
        new_points, new_gps_unique, result = get_rotation(can_coord.copy(), gps_split.copy(),
                                                          guess_orientation=guess_orientation,
                                                          guess_offest_x=guess_offest_x,
                                                          guess_offest_y=guess_offest_y,
                                                          simple=False)

        new_points.coord_x -= gps_split.iloc[0].easting
        new_points.coord_y -= gps_split.iloc[0].northing
        base_coord = new_points[["coord_x", "coord_y"]].values

        orientation, offset_x, offset_y = result.x
        offset_x += gps_split.iloc[0].easting
        offset_y += gps_split.iloc[0].northing

        loaded = True

        print("Adjust ...")
        while loopBool == True:
            time.sleep(1)

        print("Go to next map...")
        loaded = False
        loopBool = True

    print("Stop!!")


def looping_pre():  # this is new
    thread = threading.Thread(target=looping, args=())
    # thread.daemon=True   #might or might not be needed
    thread.start()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('dataset', help='Path to dataset folder.')
    arg_parser.add_argument('export_info', help='Path to export_info file.')
    arg_parser.add_argument('--map', default="util_data/upb_map_0.png", help='Map.')

    arg = arg_parser.parse_args()

    exp = arg.dataset

    # Load data
    export_info = np.load(arg.export_info).item()
    phone = pd.read_pickle("{}/phone.log.pkl".format(exp))
    steer = pd.read_csv("{}/steer.csv".format(exp))
    speed = pd.read_csv("{}/speed.csv".format(exp))

    map_viewer = ImageWgsHandler(arg.map)

    gps_unique = phone.groupby(['loc_tp']).head(1).copy()

    fig, ax = map_viewer.plot_wgs_coord(gps_unique.easting.values, gps_unique.northing.values)
    ax.set_title(f"{exp} full dataset")

    cut_intervals = export_info["cut_intervals"]
    tp_reference = export_info["tp_reference"]

    fig_path, ax_path = plt.subplots()
    fig_gps, ax_gps = plt.subplots()

    looping_pre()

    while not loaded:
        time.sleep(0.1)
    ax_gps.clear()
    map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                              ax=ax_gps)
    fig_gps.canvas.draw()
    plt.draw()

    solver_path = f"{os.path.splitext(arg.export_info)[0]}_{int(time.time()%10000)}.csv"
    print(solver_path)

    solver = open(solver_path, "a")
    solver.write(f"interval_idx,start,end,orientation,offset_x,offset_y\n")


    def press(event):
        global orientation, offset_x, offset_y, fig_path, ax_path, base_coord, gps_split, loaded
        global interval_idx

        sys.stdout.flush()
        if event.key == 'up':
            offset_y += 0.1
        elif event.key == 'down':
            offset_y -= 0.1
        elif event.key == 'right':
            offset_x += 0.1
        elif event.key == 'left':
            offset_x -= 0.1
        elif event.key == ',':
            orientation -= 0.50
        elif event.key == '.':
            orientation += 0.50
        elif event.key == 'escape':
            loaded = False

            # Save final configuration
            print(f"Done {interval_idx} - {orientation},{offset_x},{offset_y}")
            st, end = cut_intervals[interval_idx]
            st += tp_reference
            end += tp_reference
            solver.write(f"{interval_idx},{st},{end},{orientation},{offset_x},{offset_y}\n")
            solver.flush()

            closeLooping()

            while not loaded:
                time.sleep(0.1)
            print("Draw gps small")

            ax_gps.clear()
            map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                                      ax=ax_gps)
            fig_gps.canvas.draw()
            plt.draw()

        # print("Calc path with ...", orientation, offset_x, offset_y)
        new_points = get_points_rotated(base_coord, orientation, offset_x, offset_y)

        # if fig_path is not None:
        ax_path.clear()

        map_viewer.plot_wgs_coord(new_points[:, 0], new_points[:, 1], ax=ax_path)
        map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                                  ax=ax_path, show_image=False, c="b")

        plt.draw()


    fig_path.canvas.mpl_connect('key_press_event', press)
    plt.show()
    plt.pause(0.0001)
