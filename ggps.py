# Socket client example in python

import socket  # for sockets
import sys  # for exit

# create an INET, STREAMing socket
try:
except socket.error:
    print 'Failed to create socket'
    sys.exit()
print 'Socket Created'

host = 'www.google.com';
port = 80;

try:
    remote_ip = socket.gethostbyname(host)
except socket.gaierror:
    # could not resolve
    print 'Hostname could not be resolved. Exiting'
    sys.exit()

# Connect to remote server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.0.166", 50000))
s.connect(("192.168.42.129", 50000))


all = []
while True:
    reply = s.recv(4096)
    all.append(reply)
    # reply = reply.split()
    print(reply)

    # for nmea_msg in NMEA_PARSE:
    #     res = None
    #     for msg in reply:
    #         if msg.startswith(nmea_msg):
    #             res = pynmea2.parse(msg)
    #             break
    #     if res:
    #         print (res)

import pynmea2
import matplotlib.pyplot as plt
import numpy as np

all_lat = []
all_long = []
p = []
for i in range(1, 14):
    with open("/media/andrei/CE04D7C504D7AF291/nemodrive/PhonePi_SampleServer/point_" + str(i)) as f:
        t = f.readlines()

    lat = []
    long = []
    for x in t:
        for nmea_msg in ["$GPGGA"]:
            res = None
            for msg in [x]:
                if msg.startswith(nmea_msg):
                    res = pynmea2.parse(msg)
                    break
            if res:
                print (res.latitude, res.longitude)
                lat.append(res.latitude)
                long.append(res.longitude)
    all_lat.extend(lat)
    all_long.extend(long)
    p.append(dict({"lat": lat, "long": long}))


f_lat = []
f_long = []
for i in range(len(all_lat)):
    if all_lat[i] != 0:
        f_lat.append(all_lat[i])
        f_long.append(all_long[i])
first = 30
plt.scatter(f_lat[first:], f_long[first:], s=0.1)
plt.show()
m_lat = np.mean(lat[first:])
m_long = np.mean(long[first:])
res = pynmea2.parse("$GPGGA,115353.00,4426.080716,N,02602.876834,E,1,13,1.2,74.8,M,36.0,M,,*5F")

