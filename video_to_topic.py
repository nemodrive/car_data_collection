from argparse import ArgumentParser

import roslib
roslib.load_manifest('my_package')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='video_path', help='Video to read from.')
    arg_parser.add_argument(dest='topic_out', help='Topic to write to.')
    arg_parser.add_argument(dest='fps_read', default=0, help='FPS to read from video.')
    arg_parser.add_argument(dest='fps_out', default=0, help='FPS to write.')
    arg_parser.add_argument(dest='w', default=0, help='Video out width res')
    arg_parser.add_argument(dest='h', default=0, help='Video out height res')

    args = arg_parser.parse_args()

    vid = cv2.VideoCapture(args.video_path)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)

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

    frame_space = 1000.0 / float(fps_out)
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
    while ret:
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

        skip_frames += skip_frames_add

    vid.release()