#!/bin/bash
# Setup can interface (Test with candump can0)

sudo slcan_attach -f -s6 -o /dev/ttyUSB0
sudo slcand -S 1000000 ttyUSB0 can0
sudo ifconfig can0 up
