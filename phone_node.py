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
import std_msgs.msg, geometry_msgs.msg

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
        acceleration	Last measured linear acceleration of a device in three-dimensional space.covariance
        updateInterval	Sets or retrieves gyroscope interval in seconds.

    compass:
        headingAccuracy	Accuracy of heading reading in degrees.
        magneticHeading	The heading in degrees relative to the magnetic North Pole. (Read Only)
        rawVector	The raw geomagnetic data measured in microteslas. (Read Only)
        timestamp	Timestamp (in seconds since 1970) when the heading was last time updated.
        trueHeading	The heading in degrees relative to the geographic North Pole.

    acceleration	Last measured linear acceleration of a device in three-dimensional space.
 
    PhoneCollection app -> Dictionary:
        -Location
        -longitude, latitude, altitude
        Vector3 location;
        // Timestamp (in seconds since 1970) when location was last time updated
        public double loc_tp;
        // horizontal, veritcal accuracy 
        public Vector2 loc_accuracy;
    
        // -- Gyroscope
        public Quaternion attitude;
        public Vector3 gravity;
        public Vector3 rotationRate;
        public Vector3 rotationRateUnbiased;
        public float updateInterval;
        public Vector3 userAcceleration;
    
        // -- Magnetometer
        public float headingAccuracy;
        public float magneticHeading;
        public Vector3 rawVector;
        public double mag_tp;
        public float trueHeading;
    
        // -- Acceleration
        public Vector3 acceleration;
    
        public double update_tp;
}

"""


class LocalizationProcessing:
    def __init__(self):
        self.last_data = None
        self.prev_data = None
        self.origin_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.start = False
        self.last_location_update = dict({"timestamp": -1})
        self.prev_location_update = dict({"timestamp": -1})
        self.speed_from_gps = [0., 0., 0.]
        self.topic_processing = dict({
            # "gps": self.ros_get_gps,
            "imu": self.ros_get_imu,
            "magnet": self.get_magnetometer,
            "odom": self.get_odom_from_gps,
            "navsat_gps": self.get_navsat_gps
        })
        self.topic_type = dict({})

    def ingest(self, data):

        self.prev_data = self.last_data
        self.last_data = data
        if data["loc_tp"] > self.last_location_update["timestamp"]:
            self.prev_location_update = self.last_location_update

            # WGS84 conversion from lat_lon GPS
            easting, northing, zone_no, zone_letter = utm.from_latlon(
                data["location"]["x"],  data["location"]["y"])

            if not self.start:
                self.origin_position['x'] = northing
                self.origin_position['y'] = easting
                self.origin_position['z'] = data["location"]["z"]
                self.start = True

            self.last_location_update = dict({
                "timestamp": data["loc_tp"],
                "longitude": data["location"]["x"],
                "latitude": data["location"]["y"],
                "altitude": data["location"]["z"],
                "easting": easting,
                "northing": northing,
                "zone_no": zone_no,
                "zone_letter": zone_letter,
                "magnet_raw_vector": data["rawVector"],
                "quaternion": data["attitude"],
                "twist": data["rotationRateUnbiased"]
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

    def get_odom_from_gps(self):
        d = self.topic_type["odom"]()
        d.header.stamp = rospy.get_rostime()
        d.header.frame_id = "odom"
        d.child_frame_id = "base_link"

        # Position
        last_location = self.last_location_update
        d.pose.pose.position.x = last_location["northing"] - self.origin_position['x']
        d.pose.pose.position.y = last_location["easting"] - self.origin_position['y']
        d.pose.pose.position.z = last_location["altitude"] - self.origin_position['z']

        # Orientation
        gyro_attitude = last_location["quaternion"]
        d.pose.pose.orientation.x = gyro_attitude["x"]
        d.pose.pose.orientation.y = gyro_attitude["y"]
        d.pose.pose.orientation.z = gyro_attitude["z"]
        d.pose.pose.orientation.w = gyro_attitude["w"]
        d.pose.covariance = list((np.eye(6, dtype=np.float64) * 0.001).flatten())

        # speed = self.speed_from_gps
        # d.twist.twist.linear_velocity.x = speed[0]
        # d.twist.twist.linear_velocity.y = speed[1]
        # d.twist.twist.linear_velocity.z = speed[2]
        d.twist.twist.linear.x = 0
        d.twist.twist.linear.y = 0
        d.twist.twist.linear.z = 0

        # gyro_rotation = last_location["rotationRateUnbiased"]
        # d.twist.twist.angular.x = gyro_rotation["x"]
        # d.twist.twist.angular.y = gyro_rotation["y"]
        # d.twist.twist.angular.z = gyro_rotation["z"]
        d.twist.twist.angular.x = 0
        d.twist.twist.angular.y = 0
        d.twist.twist.angular.z = 0
        d.twist.covariance = list((np.eye(6, dtype=np.float64) * 0.001).flatten())

        return d

    def get_imu(self):
        data = self.last_data
        d = self.topic_type["imu"]()

        d.header.timestamp_sec = rospy.get_time()

        # TODO what is this? I do not know what to fill it with
        d.measurement_time = 0
        d.measurement_span = 0

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

    def ros_get_gps(self):
        data = self.last_data

        d = self.topic_type["gps"]()

        # Header
        d.header.stamp = rospy.get_rostime()
        d.header.frame_id = 'odom'

        # Child frame id - dont know what this is
        d.child_frame_id = 'base_link'

        # Position
        last_location = self.last_location_update
        # d.loc = geometry_msgs.msg.PoseWithCovariance()
        d.pose.pose.position.y = last_location["longitude"]
        d.pose.pose.position.x = last_location["latitude"]
        d.pose.pose.position.z = last_location["altitude"]
        d.pose.covariance = list((np.eye(6, dtype=np.float64) * 0.001).flatten())

        # Twist - linear velocity
        # speed = self.speed_from_gps
        d.twist.twist.linear.x = 0.0
        d.twist.twist.linear.y = 0.0
        d.twist.twist.linear.z = 0.0

        # Twist - angular velocity
        # gyro_rotation = data["rotationRateUnbiased"]
        d.twist.twist.angular.x = 0.0
        d.twist.twist.angular.y = 0.0
        d.twist.twist.angular.z = 0.0
        d.twist.covariance = list((np.eye(6, dtype=np.float64) * 0.001).flatten())

        return d


    def get_navsat_gps(self):
        data = self.last_data

        d = self.topic_type["navsat_gps"]()

        # Header
        d.header.stamp = rospy.get_rostime()
        d.header.frame_id = 'base_link'
        d.status.status = 0     # int8 STATUS_FIX = 0 # unaugmented fix - see sensor_msgs/NavSatStatus.msg
        d.status.service = 1    # uint16 SERVICE_GPS = 1

        last_location = self.last_location_update
        d.latitude = last_location["latitude"]
        d.longitude = last_location["longitude"]
        d.altitude = last_location["altitude"]

        d.position_covariance = list((np.eye(3, dtype=np.float64) * 0.001).flatten())
        d.position_covariance_type = 0      # uint8 COVARIANCE_TYPE_UNKNOWN = 0

        return d

    def get_magnetometer(self):
        d = self.topic_type["magnet"]()

        # Header
        d.header.stamp = rospy.get_rostime()
        d.header.frame_id = 'base_link'

        # Magnetic field
        last = self.last_location_update
        print(last["magnet_raw_vector"])
        d.magnetic_field.x = last["magnet_raw_vector"]['x']
        d.magnetic_field.y = last["magnet_raw_vector"]['y']
        d.magnetic_field.z = last["magnet_raw_vector"]['z']
        d.magnetic_field_covariance = list((np.eye(3, dtype=np.float64) * 0.05).flatten())
        return d

    def ros_get_imu(self):
        data = self.last_data
        d = self.topic_type["imu"]()

        # Header
        d.header.stamp = rospy.get_rostime()
        d.header.frame_id = 'base_link'
        # Orientation
        gyro_attitude = data["attitude"]
        d.orientation.x = gyro_attitude["x"]
        d.orientation.y = gyro_attitude["y"]
        d.orientation.z = gyro_attitude["z"]
        d.orientation.w = gyro_attitude["w"]
        d.orientation_covariance = list((np.eye(3, dtype=np.float64) * 0.05).flatten())

        # Angular velocity
        gyro_rotation = data["rotationRateUnbiased"]
        d.angular_velocity.x = gyro_rotation["x"]
        d.angular_velocity.y = gyro_rotation["y"]
        d.angular_velocity.z = gyro_rotation["z"]
        d.angular_velocity_covariance = list((np.eye(3, dtype=np.float64) * 0.05).flatten())

        # Linear acceleration
        # TODO check if necessary with or without he gravity acc
        accelerometer = data["userAcceleration"]
        d.linear_acceleration.x = accelerometer["x"]
        d.linear_acceleration.y = accelerometer["y"]
        d.linear_acceleration.z = accelerometer["z"]
        d.linear_acceleration_covariance = list((np.eye(3, dtype=np.float64) * 0.05).flatten())
        return d

    def has_topic(self, topic):
        return topic in self.topic_processing.keys()


def publish_phone(cfg):
    ip = cfg.ip
    port = cfg.port
    queue_size = cfg.queue_size
    topics = cfg.topics
    save_mode = cfg.save_mode
    simulate = cfg.simulate

    if simulate:
        save_mode = True

    if save_mode:
        import os
        import time
        save_path = cfg.save_path
        out_file_path = os.path.join(save_path, "phone_node_{}".format(time.time()))
        out_file = open(out_file_path, "w")

    rospy.init_node('publish_phone', anonymous=True)

    # Get localization class
    localization = LocalizationProcessing()

    # Generate publishers
    publishers = dict()
    for topic, topic_data in topics.items():
        topic_path, topic_type = topic_data
        print("topic_path: ", topic_path)
        print("topic_type: ", topic_type)
        if localization.has_topic(topic):
            print("topic: ", topic)
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
        # plog(message[21:-2])
        # message
        # msg_data = json.loads(message[21:-2].replace('\"\"', '\"'))
        # try:
            # print(publishers.items())
            # for (topic, publisher) in publishers.items():
            #     # print(topic)
            #     topic_data = localization.get_topic(topic)
            #     print (topic_data)
            # Ingest data point
        msg_data = json.loads(message)
        localization.ingest(msg_data)
        for (topic, publisher) in publishers.items():
            plog('cee??\n')
            print(topic)
            topic_data = localization.get_topic(topic)
            print(topic_data)
            publisher.publish(topic_data)
            # print(msg_data)
            plog("Published: {}".format(topic))
        if save_mode:
            out_file.write(message)
            out_file.write("\n")

        # except Exception:
        #     # print('fuck')
        #     pass

    # Start server:
    # run_server(ip, port, message_received, new_client, client_diconnected)
    # if not simulate:
    #     server = WebsocketServer(port, host=ip)
    #     server.set_fn_new_client(new_client)
    #     server.set_fn_client_left(client_diconnected)
    #     server.set_fn_message_received(message_received)
    #     rospy.spin()
    #
    #     server.run_forever()
    # else:

    client = dict({"id": -1})

    with open(simulate) as f:
        data_msgs = f.readlines()
        for msg in data_msgs:
            message_received(client, None, msg)


if __name__ == '__main__':
    # Config -
    cfg = argparse.Namespace()
    cfg.ip = 0
    cfg.port = 8090
    cfg.queue_size = 1
    cfg.save_mode = False
    cfg.save_path = "data/phone_node"

    cfg.simulate = ""
    #cfg.simulate = "/home/teo/nemodrive/phone_node_1536070874.9"
    cfg.simulate = "/home/alex/work/AI-MAS/projects/AutoDrive/dev/car_data_collection/data/phone_node/phone_node_1536070874.9_withEular"

    cfg.topics = dict({
        # "gps": ["/apollo/sensor/gnss/odometry", ["modules.localization.proto.gps_pb2", "Gps"]],
        # "imu": ["/apollo/sensor/gnss/imu", ["modules.drivers.gnss.proto.imu_pb2", "Imu"]]

        "imu": ["/test/imu", ['sensor_msgs.msg', 'Imu']],

        "magnet": ["/test/magnet", ['sensor_msgs.msg', 'MagneticField']],

        "odom": ["test/odom_from_gps", ['nav_msgs.msg', 'Odometry']],
        "navsat_gps": ["/test/gps", ['sensor_msgs.msg', 'NavSatFix']],
    })

    try:
        publish_phone(cfg)
    except KeyboardInterrupt:
        pass


    # # -- Local load
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import matplotlib
    # import json
    # import numpy as np
    #
    # # matplotlib.use('TkAgg')
    #
    # pd.set_option('display.height', 1000)
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    #
    # f = open("data/phone_node/phone_node_1536070874.9", "r")
    # msgs = f.readlines()
    # msgs_json = []
    # for msg in msgs:
    #     msg_data = None
    #     try:
    #         msg_data = json.loads(msg)
    #     except Exception:
    #         pass
    #
    #     if msg_data is not None:
    #         msgs_json.append(msg_data)
    #
    # df = pd.DataFrame(msgs_json)
    # df["longitude"] = df["location"].apply(lambda x: x["x"])
    # df["latitude"] = df["location"].apply(lambda x: x["y"])
    # df["altitude"] = df["location"].apply(lambda x: x["z"])
    #
    # plt.scatter(df["longitude"], df["latitude"], s=0.1)
    # plt.show()