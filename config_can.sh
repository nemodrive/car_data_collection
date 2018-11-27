#!/bin/bash
# Setup can interface (Test with candump can0)

sudo ls /dev/ttyUSB*
sudo slcan_attach -f -s6 -o /dev/ttyUSB$1
sleep 1
sudo slcand -S 1000000 ttyUSB$1 can$1
sudo ifconfig can$1 up
