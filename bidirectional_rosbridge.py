"""
    Server:
        source /opt/ros/<ros_distribution:e.g. kinetic>/setup.bash
        cd <catkin_ws folder>/src
        git clone https://github.com/RobotWebTools/rosbridge_suite.git
        cd ..
        catkin_make
        source devel/setup.bash
        pip install tornado==4.5.3
        pip install pymongo
        pip install twisted
    Client:
        'pip install ws4py' or see: https://ws4py.readthedocs.io/en/latest/sources/install/
        pip install rospy_message_converter

    Use:
        Server side:
            roslaunch rosbridge_server rosbridge_websocket.launch

        Client side:
            python bidirectional_rosbridge.py bidirectional_bridge.yaml


"""
from argparse import ArgumentParser
from json import loads, dumps
from ws4py.client.threadedclient import WebSocketClient
import rospy
import importlib
import yaml
from argparse import Namespace
import json
from rospy_message_converter import message_converter


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


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class RosbridgeClient(WebSocketClient):
    def __init__(self, *args, **kwargs):
        self.parent_class = None
        if 'parent_class' in kwargs:
            self.parent_class = kwargs.pop('parent_class')

        super(RosbridgeClient, self).__init__(*args, **kwargs)
        self.encoder = json.JSONEncoder()
        self.decoder = json.JSONDecoder()

    def opened(self):
        print("Connection opened...")

    def advertise_topic(self, topic, topic_type):
        msg = {
         'op': 'advertise',
         'topic': topic,
         'type': topic_type
        }
        self.send(dumps(msg))

    def subscribe_topic(self, topic, topic_type, queue_length=0, throttle_rate=300):
        print (queue_length, throttle_rate)
        msg = {
            'op': 'subscribe',
            "queue_length": queue_length,
            "throttle_rate": throttle_rate,
            'topic': topic,
            'type': topic_type
        }
        self.send(dumps(msg))

    def closed(self, code, reason=None):
        print(code, reason)

    def received_message(self, message):
        message = loads(message.data)
        if message["op"] == "publish":
            self.parent_class.new_message(message["topic"], message["msg"])
        else:
            print("Received weird message: {}".format(message))

    def publish(self, msg, topic):
        message = {
            'op': 'publish',
            'topic': topic,
            'msg': yaml.load(msg.__str__())
        }

        self.send(dumps(message))


class BidirectionalRosbridge:
    def __init__(self, cfg):
        server_ip = cfg.server_ip
        server_port = cfg.server_port
        self.queue_size = cfg.queue_size

        self.name = "BidirectionalRosbridge"

        ws = RosbridgeClient('ws://{}:{}/'.format(server_ip, server_port), parent_class=self)
        ws.connect()
        print ("Connected to server ...")
        self.ws = ws

        self.local_topics = cfg.local_topics
        self.local_topics_type = dict({})
        self.remote_topics = cfg.remote_topics
        self.remote_topics_type = dict({})
        self.publishers = dict({})

        rospy.init_node(self.name, anonymous=True)
        args_ = [(self.local_topics, self.local_topics_type, self.configure_local_bridge),
                (self.remote_topics, self.remote_topics_type, self.configure_remote_bridge)]

        for t, t_type, configure in args_:
            for bridge_info in t:
                que_len, throttle_rate = None, None
                if len(bridge_info) == 2:
                    topic_name, topic_type = bridge_info
                    topic_out = topic_name
                elif len(bridge_info) == 3:
                    topic_name, topic_type, topic_out = bridge_info
                elif len(bridge_info) == 5:
                    topic_name, topic_type, topic_out, que_len, throttle_rate = bridge_info
                else:
                    raise ValueError('Unknown format.')

                topic_type_class = get_class(*topic_type)
                t_type[topic_name] = topic_type_class

                if que_len is not None and throttle_rate is not None:
                    configure(topic_name, topic_type_class, topic_out,
                              queue_length=que_len, throttle_rate=throttle_rate)
                else:
                    configure(topic_name, topic_type_class, topic_out)

    def configure_local_bridge(self, topic_name, topic_type_class, topic_out,
                               queue_length=0, throttle_rate=300):
        self.ws.advertise_topic(topic_out, topic_type_class._type)
        rospy.Subscriber(topic_name, topic_type_class, self.ws.publish, topic_out)

    def configure_remote_bridge(self, topic_name, topic_type_class, topic_out,
                                queue_length=0, throttle_rate=300):
        self.ws.subscribe_topic(topic_name, topic_type_class._type, queue_length, throttle_rate)
        self.publishers[topic_name] = rospy.Publisher(topic_out,
                                                      topic_type_class,
                                                      queue_size=self.queue_size)

    def new_message(self, topic_name, msg):
        data = message_converter.convert_dictionary_to_ros_message(
            self.remote_topics_type[topic_name]._type, msg)
        self.publishers[topic_name].publish(data)

    def run(self):
        rospy.spin()

    def close(self):
        self.ws.close()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Script for syncing topics between machines via '
                                            'rosbridge websockets.')
    arg_parser.add_argument('config', help='Path to config file.')
    args = arg_parser.parse_args()

    cfg = read_cfg(args.config)
    comm = BidirectionalRosbridge(cfg)

    try:
        comm.run()
    except KeyboardInterrupt:
        comm.close()
