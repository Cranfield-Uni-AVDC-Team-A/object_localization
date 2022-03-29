#!/usr/bin/env python3

from __future__ import print_function
import os
import time
import sys
import threading

import rospy
import rospkg
pack_path = rospkg.RosPack().get_path("yolo_ros")
sys.path.append(pack_path)

import message_filters
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError

import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from yolo_trt import Yolo_TRT

from uav_stack_msgs.msg import Detector2DArray
from uav_stack_msgs.msg import Detector2D
from uav_stack_msgs.msg import Detector3DArray
from uav_stack_msgs.msg import Detector3D

from utils.helpers import *

class ObjectLocatorNode(object):
    def __init__(self):
        """Constructor"""
        self._cv_bridge = CvBridge()
        self._last_rgb_msg = None
        self._last_depth_msg = None
        self._last_cam_info = None
        self._last_location = None
        self._msg_lock =  threading.lock()
        self.read_params()
        self.init_yolo()
        self.init_pub_sub()
        self._cuda_ctx = cuda.Device(0).make_context()
        self._detector = Yolo_TRT((self.model_path + self.model), (self.h, self.w), self.category_num)
        print("[YOLO-Node] Ros Node Initialization done")
    
    def __del__(self):
        """Destructor"""
        self._cuda_ctx.pop()
        del self._detector
        del self._cuda_ctx
    
    def read_params(self):
        """ Initializes ros parameters """
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolo_ros")
        self._publish_rate = rospy.get_param("/publish_rate", 10)
        self._rgb_image_topic = rospy.get_param("/rgb_image_topic", "/dummy_rgb_image_topic")
        self._depth_image_topic = rospy.get_param("/depth_image_topic", "/dummy_depth_image_topic")
        self._cam_info_topic = rospy.get_param("/camera_info_topic", "/camera_info")
        self._self_location_topic = rospy.get_param("/self_location", "/local_position/pose")

        self._model = rospy.get_param("/model", "yolov4-416")
        self._model_path = rospy.get_param(
            "/model_path", package_path + "/models/")
        self._category_num = rospy.get_param("/category_number", 2)
        self._input_shape = rospy.get_param("/input_shape", "416")
        self._conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self._show_img = rospy.get_param("/show_image", False)
        self._namesfile = rospy.get_param("/namesfile_path", package_path+ "/cfg/obj.names")
        self._enable_depth = rospy.get_param("/locating_method/enable_depth", True)
        self._enable_pinhole = rospy.get_param("/locating_method/enable_pinhole", False)
    
    def init_pub_sub(self):
        self._rgb_image_sub = message_filters.Subscriber(self._rgb_image_topic, Image)
        self._depth_image_sub = message_filters.Subscriber(self._depth_image_topic, Image)
        self._cam_info_sub = message_filters.Subscriber(self._cam_info_topic, CameraInfo)
        self._self_location_sub =  message_filters.Subscriber(self._self_location_topic, PoseStamped)
        self._approx_time_sync = message_filters.ApproximateTimeSynchronizer([self._rgb_image_sub, self._depth_image_sub, self._cam_info_sub, self._self_location_sub], 10)
        self._approx_time_sync.registerCallback(self.camera_callback)

        self._detection_pub = rospy.Publisher("/detections", Detector2DArray, queue_size=1)
        self._information_pub = rospy.Publisher("/object_information", Detector3DArray, queue_size=1)

        self._overlay_pub = rospy.Publisher("/overlay", Image, queue_size=1)
    
    def init_yolo(self):
        """ Initialises yolo parameters required for the TensorRT engine """

        if self._model.find('-') == -1:
            self._model = self._model + "-" + self._input_shape
            
        yolo_dim = self.model.split('-')[-1] # yolo_dim = input size = 480

        self._h = self._w = int(yolo_dim)  # h = w = 480
        if self._h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self._category_num)   # cls_dict = {0: 'drone', 1: 'human'}
        self._class_names = load_class_names(self.namesfile)

        self._vis = BBoxVisualization(cls_dict)
    
    def camera_callback(self, rgb_msg, depth_msg, cam_info_msg, location_msg):
        print("[Camera Callback] Received new images")
        if self._msg_lock.acquire(False):
            self._last_rgb_msg = rgb_msg
            self._last_depth_msg = depth_msg
            self._last_cam_info = cam_info_msg
            self._last_location = location_msg
            self._msg_lock.release()
    
    def postprocess(self, clss, boxes, confs):
        objects_detected_dict = dict()
        for i in range(len(clss)):
            label = clss[i]
            label_with_num = str(label) + '_' + str(i)
            objects_detected_dict[label_with_num] = [label, boxes[i], confs[i]]
        
        return objects_detected_dict
    
    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            status = 0
            # Mutel Lock Mechanism
            if self._msg_lock.acquire(False):
                rgb_msg = self._last_rgb_msg
                self._last_rgb_msg = None

                depth_msg = self._last_depth_msg
                self._last_depth_msg = None

                cam_info = self._last_cam_info
                self._last_cam_info = None
            else:
                rate.sleep()
                continue
            
            if rgb_msg != None and depth_msg != None and cam_info != None:

                if status == 0:
                    # Convert ROS Image msg into CV Image (BGR encoding)
                    try:
                        cv_img = self._cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                        rospy.logdebug("ROS Image converted for processing")
                    except CvBridgeError as e:
                        rospy.logerr("Failed to convert image %s", str(e))

                    detections = self._detector.detect(cv_img, self.conf_th)
                    cv_img, boxes, clss, confs = plot_boxes_cv2(cv_img, detections, self._class_names)
                    
                    objects_detected_dict = dict()
                    objects_detected_dict = self.postprocess(clss, boxes, confs)  # [0_1:{1, [x, y, h, w], confs}, 1_0:{0, x, y, h, w, confs}
                    objects_list = list(objects_detected_dict.keys())  # [0_1, 1_0, ...]
                    print('Tracking the following objects', objects_list)
                    depths = dict() # {clsID_num: depth}
                    depths = self.get_object_depth(objects_list, objects_detected_dict)
                    object_locations = dict() # {clsID_num: location}
                    object_locations = self.get_object_location(objects_list, objects_detected_dict, depths)

                    desired_object = dict() #{clsID_num: {clsID,}}
                    desired_object = self.initial_filter(objects_list, objects_detected_dict, object_locations, self._last_location)

                    
                    

                    """" Multiple tracker for Monitor Drone
                    multiple_trackers = dict()
                    if len(objects_list) > 0:
                        multiple_trackers = {key: cv2.TrackerKCF_create() for key in objects_list}
                        for item in objects_list:
                            multiple_trackers[item].init(cv_img, objects_detected_dict[item][1])
                    
                        status = 1 # Detect and Track Initialization done
                    """    




                    







    
    



        



def main():
    rospy.init_node("object_locator_node")
    node = ObjectLocatorNode()
    print("[DEBUG] Node Initialization Done ~")
    node.run()

if __name__ == '__main__':
    main()