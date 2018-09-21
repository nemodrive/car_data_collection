import rospy
from nav_msgs.msg import Odometry as message_type
import pyrosbag as prb
import rosbag
from gmplot import gmplot
import matplotlib.pyplot as plt


topic_name = 'gps/filtered'
# topic_name = 'test/odom_from_gps'
# bag_name = '/home/teo/Downloads/gps-filtered-with-original-v7.bag'
bag_name ='/home/teo/car_data_collection/data/rosbags/covariance-test-v15.bag'

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber(topic_name, message_type ,callback)

    rospy.spin()

def read_bag(name):
    bag = rosbag.Bag(name)
    # gmap = gmplot.GoogleMapPlotter(44.435179, 26.047837, 18)
    lat_filtered = []
    long_filtered = []
    lat_real =[]
    long_real = []
    for topic, msg, _ in bag.read_messages():
        print(topic)
        if topic == 'gps/filtered':
            x = msg.latitude
            y = msg.longitude
            lat_filtered.append(x)
            long_filtered.append(y)
        if topic == 'test/gps':
            x = msg.latitude
            y = msg.longitude
            # print (x, y)
            # exit(0)
            lat_real.append(x)
            long_real.append(y)
    # gmap.scatter(lat_filtered, long_filtered, 'cornflowerblue', edge_width=10)
    # gmap.scatter(lat_real, long_real, 'red', edge_width=10)
    # gmap.draw('map-covariance-v3.html')
    plt.scatter(lat_filtered, long_filtered, c='b', label='Ours')
    plt.scatter(lat_real, long_real, c='r', label='GPS')
    bag.close()
    plt.show()

if __name__ == '__main__':
    read_bag(bag_name)