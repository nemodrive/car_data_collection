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

"""
from argparse import ArgumentParser
from json import dumps
from ws4py.client.threadedclient import WebSocketClient
from utils import read_cfg

class RosbridgeClient(WebSocketClient):
    def opened(self):
        print("Connection opened...")

    def advertise_topic(self, topic, topic_type):
        msg = {
         'op': 'advertise',
         'topic': topic,
         'type': topic_type
        }
        self.send(dumps(msg))

    def subscribe_topic(self, topic, topic_type):
        msg = {
            'op': 'subscribe',
            'topic': topic,
            'type': topic_type
        }
        self.send(dumps(msg))

    def closed(self, code, reason=None):
        print(code, reason)

    def received_message(self, message):
        print(message)

    def publish(self, topic, msg):
        msg = {
            'op': 'publish',
            'topic': topic,
            'msg': msg
        }

        self.send(dumps(msg))


class BidirectionalRosbridge:
    def __init__(self, cfg):
        server_ip = cfg.server_ip
        server_port = cfg.server_port
        ws = RosbridgeClient('ws://{}:{}/'.format(server_ip, server_port))
        ws.connect()
        ws.run_forever()
        print ("Connected to server ...")
        self.ws = ws

        self.local_topics = cfg.local_topics
        self.remote_topics = cfg.remote_topics

    def configure_local_bridge(self, topic_name, topic_type, topic_out):
        pass

    def configure_remote_bridge(self, topic_name, topic_type, topic_out):
        pass

    def run(self):
        pass

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
