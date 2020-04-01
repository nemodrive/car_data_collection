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
from car_utils import OFFSET_STEERING, WHEEL_STEER_RATIO

loopBool = True
orientation, offset_x, offset_y = 0., 0., 0.
fig_gps, fig_path = None, None
ax_gps, ax_path = None, None
base_coord = None
gps_split = None
speed_split, steer_split = None, None
loaded = False
interval_idx = -1
steering_offset,  steering_ratio = None, None
factor = 1.


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
    global speed_split, steer_split
    global interval_idx
    global steering_offset,  steering_ratio

    g_out_fld = export_info["g_out_fld"]
    for idx, (tp_start, tp_end) in enumerate(cut_intervals):
        if idx < start_idx:
            continue

        print(f"{idx} - {tp_start} : {tp_end}")
        interval_idx = idx

        tp_start += tp_reference
        tp_end += tp_reference

        rideID = export_info["rideIDs"][idx]
        camera_base_path = os.path.join(g_out_fld, rideID[:16])
        camera_path = f"{camera_base_path}-{0}.mov"

        if os.path.isfile(camera_path):
            vid = cv2.VideoCapture(camera_path)
            no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

            ret, frame = vid.read()
            cv2.imshow("First frame", frame)
            cv2.waitKey(1)

            vid.set(cv2.CAP_PROP_POS_FRAMES, no_frames-2)
            ret, frame = vid.read()
            cv2.imshow("Last frame", frame)
            cv2.waitKey(3)

            vid.set(cv2.CAP_PROP_POS_FRAMES, int(no_frames/3))
            ret, frame = vid.read()
            cv2.imshow("Frame 1/3", frame)
            cv2.waitKey(3)

            vid.set(cv2.CAP_PROP_POS_FRAMES, int(no_frames/3*2))
            ret, frame = vid.read()
            cv2.imshow("Frame 2/3", frame)
            cv2.waitKey(3)

        phone_split = phone[(phone.tp >= tp_start) & (phone.tp < tp_end)]
        steer_split = steer[(steer.tp >= tp_start) & (steer.tp < tp_end)]
        speed_split = speed[(speed.tp >= tp_start) & (speed.tp < tp_end)]
        gps_split = gps_unique[(gps_unique.tp >= tp_start) & (gps_unique.tp < tp_end)]

        if prev_annotation is None or idx not in prev_annotation_idx:
            steering_offset = OFFSET_STEERING
            steering_ratio = WHEEL_STEER_RATIO

            print(len(speed_split))
            print(len(steer_split))
            can_coord = get_car_can_path(speed_split.copy(), steer_split.copy())

            guess_orientation = np.random.uniform(0, 360)
            guess_offest_x = np.random.uniform(-4, 4)
            guess_offest_y = np.random.uniform(-4, 4)

            if len(gps_split) > 0:
                new_points, new_gps_unique, result = get_rotation(can_coord.copy(), gps_split.copy(),
                                                                  guess_orientation=guess_orientation,
                                                                  guess_offest_x=guess_offest_x,
                                                                  guess_offest_y=guess_offest_y,
                                                                  simple=False)
                orientation, offset_x, offset_y = result.x
                offset_x += gps_split.iloc[0].easting
                offset_y += gps_split.iloc[0].northing
            else:
                print("ERROR: No GPS Points (will random guess)")
                orientation = guess_orientation
                # offset_x, offset_y # Previous offset
                closest_gps = gps_unique.iloc[(gps_unique.tp - tp_start).values.argmin()]
                offset_x = closest_gps.easting
                offset_y = closest_gps.northing

        else:
            prev = prev_annotation[prev_annotation.interval_idx == idx].iloc[0]

            orientation = prev.orientation
            steering_offset = prev.steering_offset
            steering_ratio = prev.wheel_steer_ratio

            if use_px_coord:
                offset_px_row, offset_px_col = prev.offset_px_row, prev.offset_px_col
                offset_x, offset_y = map_viewer.get_wgs_coord(offset_px_row, offset_px_col)
                offset_x, offset_y = offset_x[0], offset_y[0]
            else:
                offset_x, offset_y = prev.offset_x, prev.offset_y

            can_coord = get_car_can_path(speed_split.copy(), steer_split.copy(), steering_offset=steering_offset,
                                         wheel_steer_ratio=steering_ratio)

        base_coord = can_coord[["coord_x", "coord_y"]].values

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
    arg_parser.add_argument('--map', default="util_data/high_res_full_UPB_hybrid.jpg", help='Map.')
    arg_parser.add_argument('--start-idx', default=0, type=int, help='interval start idx.')
    arg_parser.add_argument('--prev-annotation', default=None, type=str, help='Previous annotation csv file.')
    arg_parser.add_argument('--use-px-coord', default=False, type=bool, help='Use conversion from px to coord from '
                                                                            'csv file.')

    arg = arg_parser.parse_args()
    start_idx = arg.start_idx
    exp = arg.dataset
    prev_annotation = arg.prev_annotation
    use_px_coord = arg.use_px_coord
    prev_annotation_idx = []

    # Load data
    export_info = np.load(arg.export_info).item()
    phone = pd.read_pickle("{}/phone.log.pkl".format(exp))
    steer = pd.read_csv("{}/steer.csv".format(exp))
    speed = pd.read_csv("{}/speed.csv".format(exp))
    map_viewer = ImageWgsHandler(arg.map)

    if prev_annotation:
        prev_annotation = pd.read_csv(prev_annotation)
        prev_annotation_idx = prev_annotation.interval_idx.values

    gps_unique = phone.groupby(['loc_tp']).head(1).copy()

    fig, ax = map_viewer.plot_wgs_coord(gps_unique.easting.values, gps_unique.northing.values)
    ax.set_title(f"{exp} full dataset")

    cut_intervals = export_info["cut_intervals"]
    tp_reference = export_info["tp_reference"]

    fig_path, ax_path = plt.subplots()
    fig_gps, ax_gps = plt.subplots()

    looping_pre()
    convert_method = 1

    while not loaded:
        time.sleep(0.1)
    ax_gps.clear()
    map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                              ax=ax_gps)
    fig_gps.canvas.draw()
    plt.draw()

    solver_path = f"{os.path.splitext(arg.export_info)[0]}_{int(time.time()%10000)}.csv"
    print(f"Writing solutions to: {solver_path}")

    solver = open(solver_path, "a")
    solver.write(f"interval_idx,start,end,orientation,offset_x,offset_y,"
                 f"offset_px_row,offset_px_col,steering_offset,wheel_steer_ratio"
                 f"\n")


    def press(event):
        global orientation, offset_x, offset_y, fig_path, ax_path, base_coord, gps_split, loaded
        global interval_idx
        global convert_method
        global steering_offset, steering_ratio
        global speed_split, steer_split
        global factor

        load_next, save_conf = False, False
        redo_can_path = False

        sys.stdout.flush()
        if event.key == 'up':
            offset_y += 0.1 * factor
        elif event.key == 'down':
            offset_y -= 0.1 * factor
        elif event.key == 'right':
            offset_x += 0.1 * factor
        elif event.key == 'left':
            offset_x -= 0.1 * factor
        elif event.key == ',':
            orientation -= 0.30 * factor
        elif event.key == '.':
            orientation += 0.30 * factor
        elif event.key == 'u':
            redo_can_path = True
            steering_offset += 0.10 * factor
        elif event.key == 'i':
            redo_can_path = True
            steering_offset -= 0.10 * factor
        elif event.key == 'o':
            redo_can_path = True
            steering_ratio += 0.05 * factor
        elif event.key == 'p':
            redo_can_path = True
            steering_ratio -= 0.05 * factor
        elif event.key == '-':
            redo_can_path = True
            factor *= 0.1
            print(f"New factor: {factor}")
        elif event.key == '+':
            redo_can_path = True
            factor *= 10.
            print(f"New factor: {factor}")
        elif event.key == 'c':
            convert_method = int(not convert_method)
        elif event.key == 'n':
            load_next = True
            save_conf = True
        elif event.key == 'm':
            print(f"This {interval_idx} - {orientation},{offset_x},{offset_y},{steering_offset},{steering_ratio}")
        elif event.key == 'x':
            print(f"skip interval: {interval_idx}")
            save_conf = False
            load_next = True
        elif event.key == 'e':
            plt.close("all")
            cv2.destroyAllWindows()
            exit(0)

        if redo_can_path:
            can_coord = get_car_can_path(speed_split.copy(), steer_split.copy(), steering_offset=steering_offset,
                                         wheel_steer_ratio=steering_ratio)
            base_coord = can_coord[["coord_x", "coord_y"]].values

        if load_next:
            loaded = False

            if save_conf:
                # Save final configuration
                print(f"Done {interval_idx} - {orientation},{offset_x},{offset_y},{steering_offset},{steering_ratio}")
                st, end = cut_intervals[interval_idx]
                st += tp_reference
                end += tp_reference

                px_row, px_col = map_viewer.get_image_coord([offset_x], [offset_y])

                solver.write(f"{interval_idx},{st},{end},{orientation},{offset_x},{offset_y},"
                             f"{px_row[0]},{px_col[0]},{steering_offset},{steering_ratio}\n")
                solver.flush()

            closeLooping()

            while not loaded:
                time.sleep(0.1)
            print("Draw gps small")

            ax_gps.clear()
            map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                                      ax=ax_gps, convert_method=convert_method)
            fig_gps.canvas.draw()
            plt.draw()

            steering_offset = OFFSET_STEERING
            steering_ratio = WHEEL_STEER_RATIO

        new_points = get_points_rotated(base_coord, orientation, offset_x, offset_y)

        # if fig_path is not None:
        ax_path.clear()

        map_viewer.plot_wgs_coord(new_points[:, 0], new_points[:, 1], ax=ax_path, convert_method=convert_method)
        map_viewer.plot_wgs_coord(gps_split.easting.values, gps_split.northing.values,
                                  ax=ax_path, show_image=False, c="b", convert_method=convert_method)

        plt.draw()


    fig_path.canvas.mpl_connect('key_press_event', press)
    plt.show()
    plt.pause(0.0001)
