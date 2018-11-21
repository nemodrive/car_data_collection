from argparse import ArgumentParser
import rospy

from modules.drivers.proto.conti_radar_pb2 import ContiRadar


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Empty topic.')
    args = arg_parser.parse_args()

    rospy.init_node("Empty topics", anonymous=True)
    rospy.Publisher("/apollo/sensor/conti_radar",
                    topic_type_class,
                    queue_size=self.queue_size)
    try:
    except KeyboardInterrupt:
