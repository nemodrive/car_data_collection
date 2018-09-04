#!/usr/bin/env python
"""
    Install
        pip install utm, websocket_server
"""
import rospy
import json
import argparse
from websocket_server import WebsocketServer
import utm
from math import radians, cos, sin, asin, sqrt
import numpy as np
import importlib

from modules.localization.proto import gps_pb2
from modules.localization.proto import imu_pb2

LOG = True


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers is 6371
    m = 6371 * 1000 * c
    return m


def plog(s):
    if LOG:
        print('[Phone] {}'.format(s))


def get_local_ip():
    """Try to get local ip used for internet connection"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    found_ip = s.getsockname()[0]
    s.close()

    return found_ip


def new_client(client, server):
    """Called for every client connecting (after handshake)"""
    plog("New client connected with id {} and address {}, {}".format(client['id'],
                                                                     client['address'][0],
                                                                     client['address'][1]))


def client_diconnected(client, server):
    """Called for every client disconnecting"""
    plog("Client(%d) disconnected" % client['id'])


def std_message_received(client, server, message):
    """Called when a client sends a message"""
    # plog("Client(%d) said: %s" % (client['id'], message))


def run_server_classic(ip_address, port):
    """run_server"""
    server = WebsocketServer(port, host=ip_address)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_diconnected)
    server.set_fn_message_received(std_message_received)
    server.run_forever()


def run_server(i_ip_address, i_port, i_message_received, i_new_client, i_client_diconnected):
    """run_server"""
    server = WebsocketServer(i_port, host=i_ip_address)
    server.set_fn_new_client(i_new_client)
    server.set_fn_client_left(i_client_diconnected)
    server.set_fn_message_received(i_message_received)
    server.run_forever()


"""
    PhoneCollection app -> 

    Location
        altitude	Geographical device location altitude.
        horizontalAccuracy	Horizontal accuracy of the location.
        latitude	Geographical device location latitude.
        longitude	Geographical device location latitude.
        timestamp	Timestamp (in seconds since 1970) when location was last time updated.
        verticalAccuracy	Vertical accuracy of the location.

    Gyro: 
        attitude	Returns the attitude (ie, orientation in space) of the device.
        gravity	Returns the gravity acceleration vector expressed in the device's reference frame.
        rotationRate	Returns rotation rate as measured by the device's gyroscope.
        rotationRateUnbiased	Returns unbiased rotation rate as measured by the device's gyroscope.
        userAcceleration	Returns the acceleration that the user is giving to the device.
        acceleration	Last measured linear acceleration of a device in three-dimensional space. 

    compass:
        headingAccuracy	Accuracy of heading reading in degrees.
        magneticHeading	The heading in degrees relative to the magnetic North Pole. (Read Only)
        rawVector	The raw geomagnetic data measured in microteslas. (Read Only)
        timestamp	Timestamp (in seconds since 1970) when the heading was last time updated. 
        trueHeading	The heading in degrees relative to the geographic North Pole. 

"""


class LocalizationProcessing:
    def __init__(self):
        self.last_data = None
        self.prev_data = None
        self.last_location_update = dict({"timestamp": -1})
        self.prev_location_update = dict({"timestamp": -1})
        self.speed_from_gps = [0., 0., 0.]
        self.topic_processing = dict({
            "gps": self.get_gps,
            "imu": self.get_imu
        })
        self.topic_type = dict({})

    def ingest(self, data):
        self.prev_data = self.last_data
        self.last_data = data
        if data["loc_tp"] > self.last_location_update["timestamp"]:
            self.prev_location_update = self.last_location_update

            easting, northing, zone_no, zone_letter = utm.from_latlon(
                data["location"]["x"],  data["location"]["y"])

            self.last_location_update = dict({
                "timestamp": data["loc_tp"],
                "longitude": data["location"]["x"],
                "latitude": data["location"]["y"],
                "altitude": data["location"]["z"],
                "easting": easting,
                "northing": northing,
                "zone_no": zone_no,
                "zone_letter": zone_letter
            })

            last_location = self.last_location_update
            prev_location = self.prev_location_update
            if len(prev_location) > 1:
                self.speed_from_gps = np.array([
                    last_location[x] - prev_location[x]
                    for x in ["easting", "northing", "altitude"]])
                self.speed_from_gps = abs(self.speed_from_gps) / \
                                      float(data["loc_tp"] - prev_location["timestamp"]) * 1000

    def get_topic(self, topic):
        data = self.topic_processing[topic]()
        return data

    def register_topic_type(self, topic, topic_type):
        self.topic_type[topic] = topic_type

    def get_gps(self):
        data = self.last_data

        d = self.topic_type["gps"]()

        d.header.timestamp_sec = rospy.get_time()

        # Position
        last_location = self.last_location_update
        d.localization.position.x = last_location["easting"]
        d.localization.position.y = last_location["northing"]
        d.localization.position.z = last_location["altitude"]

        # Orientation
        gyro_attitude = data["attitude"]
        d.localization.orientation.qx = gyro_attitude["x"]
        d.localization.orientation.qy = gyro_attitude["y"]
        d.localization.orientation.qz = gyro_attitude["z"]
        d.localization.orientation.qw = gyro_attitude["w"]

        speed = self.speed_from_gps
        d.localization.linear_velocity.x = speed[0]
        d.localization.linear_velocity.y = speed[1]
        d.localization.linear_velocity.z = speed[2]
        return d

    def get_imu(self):
        data = self.last_data
        d = self.topic_type["imu"]()

        d.header.timestamp_sec = rospy.get_time()
        d.measurement_time = rospy.get_time()
        d.measurement_span = rospy.get_time()

        # Linear acceleration
        # TODO check if necessary with or without he gravity acc
        accelerometer = data["userAcceleration"]

        d.linear_acceleration.x = accelerometer["x"]
        d.linear_acceleration.y = accelerometer["y"]
        d.linear_acceleration.z = accelerometer["z"]

        # Angular velocity
        gyro_rotation = data["rotationRateUnbiased"]
        d.angular_velocity.x = gyro_rotation["x"]
        d.angular_velocity.y = gyro_rotation["y"]
        d.angular_velocity.z = gyro_rotation["z"]
        return d

    def has_topic(self, topic):
        return topic in self.topic_processing.keys()


def publish_phone(cfg):
    ip = cfg.ip
    port = cfg.port
    queue_size = cfg.queue_size
    topics = cfg.topics

    rospy.init_node('publish_phone', anonymous=True)

    # Get localization class
    localization = LocalizationProcessing()

    # Generate publishers
    publishers = dict()
    for topic, topic_data in topics.items():
        topic_path, topic_type = topic_data
        if localization.has_topic(topic):
            topic_class = get_class(*topic_type)
            localization.register_topic_type(topic, topic_class)
            publishers[topic] = rospy.Publisher(topic_path, topic_class, queue_size=queue_size)
            plog("Registered {}".format(topic))

    if ip == 0:
        ip = get_local_ip()
        plog("Found ip: {}".format(ip))

    plog("Star server on: {}:{}".format(ip, port))

    # Configure message received
    def message_received(client, server, message):
        """Called when a client sends a message"""
        plog("Client(%d) said: %s" % (client['id'], message))

        # message
        try:
            msg_data = json.loads(message)

            # Ingest data point
            localization.ingest(msg_data)
            for topic, publisher in publishers.items():
                topic_data = localization.get_topic(topic)
                publisher.publish(topic_data)
                plog("Published: {}".format(topic))
        except Exception:
            pass

    # Start server:
    # run_server(ip, port, message_received, new_client, client_diconnected)
    server = WebsocketServer(port, host=ip)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_diconnected)
    server.set_fn_message_received(message_received)
    server.run_forever()

    rospy.spin()


if __name__ == '__main__':
    # Config -
    cfg = argparse.Namespace()
    cfg.ip = 0
    cfg.port = 8090
    cfg.queue_size = 1
    cfg.topics = dict({
        "gps": ["/apollo/sensor/gnss/odometry", ["modules.localization.proto.gps_pb2", "Gps"]],
        "imu": ["/apollo/sensor/gnss/imu", ["modules.drivers.gnss.proto.imu_pb2", "Imu"]]
    })

    try:
        publish_phone(cfg)
    except KeyboardInterrupt:
        pass
