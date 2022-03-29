#!/usr/bin/env python3

from __future__ import print_function
import os
import time
import sys

import rospy
import rospkg
pack_path = rospkg.RosPack().get_path("object_localization")
sys.path.append(pack_path)

from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError


import numpy as np

from uav_stack_msgs.msg import Detector2DArray
from uav_stack_msgs.msg import Detector2D
from uav_stack_msgs.msg import Detector3DArray
from uav_stack_msgs.msg import Detector3D



class ROS_Timer(object):
    def __init__(self):
        self.last_rec_time = 0
        self.new_rec_time = 0
        rospy.Subscriber("/detections", Detector3DArray, self.callback)
        
    def callback(self, msg):
        self.new_rec_time = time.time()
        time_passed = self.new_rec_time - self.last_rec_time
        #fps = 1.0 / time_passed
        #print("FPS = {}".format(fps))
        print("Time passed = {}".format(time_passed))
        self.last_rec_time = self.new_rec_time
    


def main():
    rospy.init_node("subscriber_test")
    node = ROS_Timer()
    rospy.spin()


if __name__ == "__main__":
    main()
