# Andrei, 2018
"""
    Interactive script to playback video to ros specified ros topic.
"""

from argparse import ArgumentParser
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import sys

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument('video_path', help='Video to read from.')
    arg_parser.add_argument('topic_out', help='Topic to write to.')
    arg_parser.add_argument('--fps_read', default=0, help='What FPS to read from video at.')
    arg_parser.add_argument('--fps_out', default=0, help='What FPS to write at.')
    arg_parser.add_argument('--w', default=0, help='Video out width res')
    arg_parser.add_argument('--h', default=0, help='Video out height res')
    arg_parser.add_argument('--wait_key', dest='wait_key', action='store_true',
                            help='Wait for key between frames')
    arg_parser.add_argument('--start_pos', default=0.,
                            help='Pos in video to start at ( in fraction [0.,1) )')

    args = arg_parser.parse_args()
    wait_key = args.wait_key
    start_pos = args.start_pos

    vid = cv2.VideoCapture(args.video_path)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    if start_pos != 0.:
        vid_start_pos = int(total_frames * start_pos)
        vid.set(cv2.CAP_PROP_POS_FRAMES, vid_start_pos)
        print("Video has {} FRAMES and start position set at FRAME_NO: {}".format(total_frames,
                                                                                  vid_start_pos))

    # Determine video original fps
    if vid_fps <= 0:
        print("Cannot read fps with CAP_PROP_FPS")
        vid_fps = float(input("What is the video fps?"))

    print("Frames per second of video: {0}".format(vid_fps))

    fps_out = args.fps_out if args.fps_out != 0 else vid_fps

    # Determine if will skip frames when reading from video
    skip_frames_add = 1.
    if args.fps_read != 0:
        skip_frames_add = vid_fps / float(args.fps_read)

    frame_space = 1.0 / float(fps_out)
    skip_frames = 1.

    # Read first frame
    ret, frame = vid.read()
    orig_h, orig_w, _ = frame.shape
    out_w = orig_w if args.w == 0 else args.w
    out_h = orig_h if args.h == 0 else args.h

    # Start publisher
    rospy.init_node('image_converter', anonymous=True)
    image_pub = rospy.Publisher(args.topic_out, Image)
    bridge = CvBridge()

    last_show_tp = 0

    print("")
    print("Start show ...")
    frame_cnt = 0
    while ret and not rospy.is_shutdown():
        show_frame = cv2.resize(frame, (out_w, out_h))
        try:
            wait_time = max(0., frame_space - (time.time() - last_show_tp))
            time.sleep(wait_time)
            image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
            last_show_tp = time.time()
        except CvBridgeError as e:
            print(e)

        while skip_frames >= 1.:
            ret, frame = vid.read()
            skip_frames -= 1.
            frame_cnt += 1

        skip_frames += skip_frames_add
        if frame_cnt % vid_fps == 0:
            sys.stdout.write("\rPos in video: {:0.4f}".format(frame_cnt/float(total_frames)))
            sys.stdout.flush()

    vid.release()