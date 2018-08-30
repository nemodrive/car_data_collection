#!/usr/bin/env python
import rospy
import json
import argparse
from websocket_server import WebsocketServer
import utm
from math import radians, cos, sin, asin, sqrt

from modules.localization.proto import gps_pb2
from modules.localization.proto import imu_pb2

LOG = False


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

    def ingest(self, data):
        self.prev_data = self.last_data
        self.last_data = data
        if data["location"]["timestamp"] > self.last_location_update["timestamp"]:
            self.prev_location_update = self.last_location_update
            self.last_location_update = data["location"]


        pass

    def get_topic(self, topic):
        pass

    def get_gps(self, data):
        d = gps_pb2.Gps()

        d.header.timestamp_sec = rospy.get_time()

        # Position
        gps = data["gps"]
        wgs = utm.from_latlon(gps["x"], gps["y"])
        d.localization.position = dict({
            "x": wgs[0],
            "y": wgs[1],
            "z": wgs[2],
        })

        # Orientation
        gyro_attitude = data["gyro_attitude"]
        d.localization.orientation = dict({
            "qx": gyro_attitude["x"],
            "qy": gyro_attitude["y"],
            "qz": gyro_attitude["z"],
            "qw": gyro_attitude["w"],
        })

        return d

    def get_imu(self, data):
        d = imu_pb2.CorrectedImu()

        d.header.timestamp_sec = rospy.get_time()
        d.measurement_time = rospy.get_time()
        d.measurement_span = rospy.get_time()

        # Linear acceleration
        # TODO check if necessary with or without he gravity acc
        accelerometer = data["accelerometer"]
        d.linear_acceleration = dict({
            "x": accelerometer["x"],
            "y": accelerometer["y"],
            "z": accelerometer["z"],
        })

        # Angular velocity
        gyro_rotation = data["gyro_rotation"]
        d.angular_velocity = dict({
            "x": gyro_rotation["x"],
            "y": gyro_rotation["y"],
            "z": gyro_rotation["z"],
        })

    def has_topic(self, topic):
        return False


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
    for topic, (topic_path, topic_type) in topics.items:
        if localization.has_topic(topic):
            publishers[topic] = rospy.Publisher(topic_path, topic_type, queue_size=queue_size)

    if ip == 0:
        ip = get_local_ip()
        plog("Found ip: {}".format(ip))

    plog("Star server on: {}:{}".format(ip, port))

    # Configure message received
    def message_received(client, server, message):
        """Called when a client sends a message"""
        plog("Client(%d) said: %s" % (client['id'], message))

        # message
        msg_data = json.loads(message)

        # Ingest data point
        localization.ingest(msg_data)
        for topic, publisher in publishers.items():
            topic_data = localization.get_topic(topic)
            publisher.publish(topic_data)

    # Start server:
    run_server(ip, port, message_received, new_client, client_diconnected)


if __name__ == '__main__':
    cfg = argparse.Namespace()
    cfg.ip = 0
    cfg.port = 8081
    cfg.queue_size = 1
    cfg.topics = dict({
        "gps": ("/apollo/sensor/gnss/odometry", gps_pb2.Gps),
        "imu": ("/apollo/sensor/gnss/imu", imu_pb2.CorrectedImu)
    })

    try:
        publish_phone()
    except rospy.ROSInterruptException:
        pass

    import numpy as np

    # Test gpst ->
    with open("data/odo_imu_demo_2.5") as f:
        data = json.load(f)

    odo = data['/apollo/sensor/gnss/odometry']

    points = []
    tps = []
    speed = []
    for i in range(10):
        lat, lon = utm.to_latlon(odo[i]["localization"]["position"]["x"],
                                 odo[i]["localization"]["position"]["y"], 32, "C")
        x, y, z = pm.geodetic2ecef(odo[i]["localization"]["position"]["x"],
                                   odo[i]["localization"]["position"]["y"],
                                   odo[i]["localization"]["position"]["z"])
        points.append(np.array([x,y,z]))
        tps.append(odo[i]["timestamp"])

        if i > 0:
            speed.append((points[-1] - points[-2]) / (tps[-1] - tps[-2]))

        x, y, z = pm.geodetic2ecef()
