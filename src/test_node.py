#!/usr/bin/env python3

from __future__ import print_function
from operator import mul
import os
import time
import sys
import threading

import rospy
import rospkg
pack_path = rospkg.RosPack().get_path("object_localization")
sys.path.append(pack_path)

import message_filters
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError


import numpy as np
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
from utils.helpers import extract_yolo_detections

class ObjectLocatorNode(object):
    def __init__(self):
        """Constructor"""
        self._cv_bridge = CvBridge()
        self._last_rgb_msg = None
        self._last_depth_msg = None
        self._last_cam_info = None
        #self._last_location = None
        self._rgb_msg_lock =  threading.Lock()
        self._depth_msg_lock = threading.Lock()
        self._cam_info_lock = threading.Lock()
        self.read_params()
        self.init_yolo()
        self.init_pub_sub()
        self._cuda_ctx = cuda.Device(0).make_context()
        self._detector = Yolo_TRT((self._model_path + self._model), (self._h, self._w), self._category_num)
        print("[YOLO-Node] Ros Node Initialization done")
    
    def __del__(self):
        """Destructor"""
        self._cuda_ctx.pop()
        del self._detector
        del self._cuda_ctx
    
    def read_params(self):
        """ Initializes ros parameters """
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("object_localization")
        self._publish_rate = rospy.get_param("/publish_rate", 10)
        self._rgb_image_topic = rospy.get_param("/rgb_image_topic", "/zed2/zed_node/rgb/image_rect_color")
        self._depth_image_topic = rospy.get_param("/depth_image_topic", "/zed2/zed_node/depth/depth_registered")
        self._cam_info_topic = rospy.get_param("/camera_info_topic", "/zed2/zed_node/depth/camera_info")
        #self._self_location_topic = rospy.get_param("/self_location", "/local_position/pose")

        self._model = rospy.get_param("/model", "yolov4-608")
        self._model_path = rospy.get_param(
            "/model_path", package_path + "/models/")
        self._category_num = rospy.get_param("/category_number", 80)
        self._input_shape = rospy.get_param("/input_shape", "608")
        self._conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self._show_img = rospy.get_param("/show_image", False)
        self._namesfile = rospy.get_param("/namesfile_path", package_path+ "/configs/coco.names")
        self._enable_depth = rospy.get_param("/locating_method/enable_depth", True)
        self._enable_pinhole = rospy.get_param("/locating_method/enable_pinhole", False)
    
    def init_pub_sub(self):
        self._rgb_image_sub = message_filters.Subscriber(self._rgb_image_topic, Image)
        self._depth_image_sub = message_filters.Subscriber(self._depth_image_topic, Image)
        self._cam_info_sub = message_filters.Subscriber(self._cam_info_topic, CameraInfo)
        #self._self_location_sub =  message_filters.Subscriber(self._self_location_topic, PoseStamped)
        #self._approx_time_sync = message_filters.ApproximateTimeSynchronizer([self._rgb_image_sub, self._depth_image_sub, self._cam_info_sub, self._self_location_sub], 10)
        self._approx_time_sync = message_filters.ApproximateTimeSynchronizer([self._rgb_image_sub, self._depth_image_sub, self._cam_info_sub], 10, 0.1, allow_headerless=True)
        self._approx_time_sync.registerCallback(self.camera_callback)

        self._detection_pub = rospy.Publisher("/detections", Detector3DArray, queue_size=1)
        self._information_pub = rospy.Publisher("/object_information", Detector3DArray, queue_size=1)

        self._overlay_pub = rospy.Publisher("/overlay", Image, queue_size=1)
    
    def init_yolo(self):
        """ Initialises yolo parameters required for the TensorRT engine """

        if self._model.find('-') == -1:
            self._model = self._model + "-" + self._input_shape
            
        yolo_dim = self._model.split('-')[-1] # yolo_dim = input size = 480

        self._h = self._w = int(yolo_dim)  # h = w = 480
        if self._h % 32 != 0 or self._w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self._category_num)   # cls_dict = {0: 'drone', 1: 'human'}
        self._class_names = load_class_names(self._namesfile)

        self._vis = BBoxVisualization(cls_dict)
    
    def camera_callback(self, rgb_msg, depth_msg, cam_info_msg):#, location_msg):
        print("[Camera Callback] Received new images")
        #if self._rgb_msg_lock.acquire(False):
        self._last_rgb_msg = rgb_msg
        #    self._rgb_msg_lock.release()

        #if self._depth_msg_lock.acquire(False):
        self._last_depth_msg = depth_msg
        #    self._depth_msg_lock.release()

        #if self._cam_info_lock.acquire(False):
        self._last_cam_info = cam_info_msg
            #self._last_location = location_msg
        #    self._cam_info_lock.release()
    
    def get_object_location(self, objects_detected_dict, depth_msg, cam_info):
        object_locations = dict()
        # Convert ROS Image msg into CV Image (32FC1 encoding)
        
        try:
            depth_img = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            # Convert the depth image to a Numpy array
            depth_array = np.array(depth_img, dtype=np.float32)
            rospy.logdebug("ROS Depth Image converted for processing")

        except CvBridgeError as e:
            rospy.logerr("Failed to convert image %s", str(e))
        
        cam_info_cx = cam_info.K[2]
        cam_info_cy = cam_info.K[5]
        cam_info_fx = cam_info.K[0]
        cam_info_fy = cam_info.K[4]
        print("[Camera Info Debug] cx = {} | cy = {} | fx = {} | fy = {}".format(cam_info_cx, cam_info_cy, cam_info_fx, cam_info_fy))

        for obj_key, info in objects_detected_dict.items():
            bounding_box = info[1]
            center_x = int(bounding_box[0] + (bounding_box[2]-bounding_box[0])/2)
            center_y = int(bounding_box[1] + (bounding_box[3]-bounding_box[1])/2)

            # Linear index of the center pixel
            #centerIdx = int(center_x + depth_msg.width * center_y)
            obj_depth = depth_array[center_y, center_x]
            location_x = obj_depth * ((center_x - cam_info_cx) / cam_info_fx)
            location_y = obj_depth * ((center_y - cam_info_cy) / cam_info_fy)
            location_z = obj_depth
            object_locations[obj_key] = [location_x, location_y, location_z]


        return object_locations
    
    def postprocess(self, clss, boxes, confs):
        objects_detected_dict = dict()
        for i in range(len(clss)):
            label = clss[i]
            label_with_num = str(label) + '_' + str(i)
            objects_detected_dict[label_with_num] = [label, boxes[i], confs[i]]
        
        return objects_detected_dict

    def drawPred(self,cv_img, objects_detected_dict):
        for obj_key, info in objects_detected_dict.items():
            box = info[1]
            confidence = info[2]
            label = '%s: %.2f' % (obj_key, confidence)
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            overlay_img = cv2.rectangle(cv_img, p1, p2, (0, 255, 0))
            left = int(box[0])
            top = int(box[1])
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            overlay_img = cv2.rectangle(cv_img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            overlay_img = cv2.putText(cv_img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
        return overlay_img
    
    def publisher(self, objects_detected_dict, object_locations):
        detections_msg = Detector3DArray()
        detections_msg.header.stamp = rospy.Time.now()
        detections_msg.header.frame_id = "camera"
        time_stamp = rospy.Time.now()

        for obj_key, info in objects_detected_dict.items():
            detection = Detector3D()
            detection.header.stamp=time_stamp
            detection.header.frame_id = "camera"
            detection.results.id = info[0]
            detection.results.score = info[2]
            # bbox[0] -> xmin, bbox[1] -> ymin, bbox[2] -> xmax, bbox[3] -> ymax, 
            bounding_box = info[1]
            detection.bbox.center.x = bounding_box[0] + (bounding_box[2]-bounding_box[0])/2
            detection.bbox.center.y = bounding_box[1] + (bounding_box[3]-bounding_box[1])/2
            detection.bbox.center.theta = 0.0  # change if required

            detection.bbox.size_x = abs(bounding_box[0]-bounding_box[2])
            detection.bbox.size_y = abs(bounding_box[1]-bounding_box[3])

            detection.positions.x = object_locations[obj_key][0]
            detection.positions.y = object_locations[obj_key][1]
            detection.positions.z = object_locations[obj_key][2]


            detections_msg.detections.append(detection)

            rospy.logdebug("Number of detections in the list: {}".format(len(detections_msg.detections)))
        
        self._detection_pub.publish(detections_msg)

            

    
    def run(self):
        rate = rospy.Rate(self._publish_rate)
        print("[Node.Run()] Start running!")
        status = 0
        while not rospy.is_shutdown():

            """
            # Mutel Lock Mechanism
            if self._rgb_msg_lock.acquire(False):
                rgb_msg = self._last_rgb_msg
                if rgb_msg is None:
                    print("rgb is none")
                self._last_rgb_msg = None
                self._rgb_msg_lock.release()
            
            if self._depth_msg_lock.acquire(False):
                depth_msg = self._last_depth_msg
                if depth_msg is None:
                    print("depth is none")
                self._last_depth_msg = None
                self._depth_msg_lock.release()

            if self._cam_info_lock.acquire(False):
                cam_info = self._last_cam_info
                if cam_info is None:
                    print("cam info is none")
                self._last_cam_info = None
                self._cam_info_lock.release()

            else:
                rate.sleep()
                continue
            """
            rgb_msg = self._last_rgb_msg
            self._last_rgb_msg = None

            depth_msg = self._last_depth_msg
            self._last_depth_msg = None

            cam_info = self._last_cam_info
            self._last_cam_info = None
            if rgb_msg != None and depth_msg != None and cam_info != None:

                if status == 0:
                    print("~~~~~~~~Status = 0")
                    # Convert ROS Image msg into CV Image (BGR encoding)
                    try:
                        cv_img = self._cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                        rospy.logdebug("ROS Image converted for processing")
                    except CvBridgeError as e:
                        rospy.logerr("Failed to convert image %s", str(e))

                    detections = self._detector.detect(cv_img, self._conf_th)
                    boxes, clss, confs = extract_yolo_detections(cv_img, detections, self._class_names)
                    
                    objects_detected_dict = dict()
                    objects_detected_dict = self.postprocess(clss, boxes, confs)  # [0_1:{1, [x, y, h, w], confs}, 1_0:{0, x, y, h, w, confs}
                    objects_list = list(objects_detected_dict.keys())  # [0_1, 1_0, ...]
                    print('Tracking the following objects', objects_list)
                    """
                    depths = dict() # {clsID_num: depth}
                    depths = self.get_object_depth(objects_list, objects_detected_dict)
                    object_locations = dict() # {clsID_num: location}
                    object_locations = self.get_object_location(objects_list, objects_detected_dict, depths)

                    desired_object = dict() #{clsID_num: {clsID,}}
                    desired_object = self.initial_filter(objects_list, objects_detected_dict, object_locations, self._last_location)
                    """
                    
                    

                    #Multiple tracker for Monitor Drone
                    multiple_trackers = dict()
                    if len(objects_list) > 0: # If detect success
                        multiple_trackers = {key: cv2.TrackerKCF_create() for key in objects_list}
                        for item in objects_list:
                            multiple_trackers[item].init(cv_img, objects_detected_dict[item][1])
                    
                        status = 1 # Detect and Track Initialization done
                        print("[Initialization stage] Multi-trackers are initialised")
                    else:
                        print("[Initialization stage] Dections fail --> Stay at Initialization stage")
                        status = 0
                
                elif status == 1:
                    
                    print("[Loop Stage] Tracking")
                    timer = cv2.getTickCount()
                    if len(objects_detected_dict) > 0:
                        del_items = []
                        for obj_key, tracker in multiple_trackers.items():
                            ok, bbox = tracker.update(cv_img)
                            if ok:
                                objects_detected_dict[obj_key][1] = bbox
                            else:
                                print("Tracker of #{} Lose Track".format(obj_key))
                                del_items.append(obj_key)
                        
                        for lost_track_obj_key in del_items:
                            multiple_trackers.pop(lost_track_obj_key)
                            objects_detected_dict.pop(lost_track_obj_key)
                        
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                    if len(objects_detected_dict) > 0: # meaning that not all trackers are failed
                        # the objs in the dict are the remaining tracker target
                        objects_list = list(objects_detected_dict.keys())
                        object_locations = dict() # {clsID_num: [x, y, z]}
                        object_locations = self.get_object_location(objects_detected_dict, depth_msg, cam_info)

                        # Publish the Object Information
                        self.publisher(objects_detected_dict, object_locations)

                        # Overlay Images
                        img = self.drawPred(cv_img, objects_detected_dict)
                        # Convert CV Image back to ROS Image msg for publishing the overlay image
                        try:
                            overlay_img = self._cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")
                            rospy.logdebug("CV Image converted for publishing")
                            self._overlay_pub.publish(overlay_img)
                        except CvBridgeError as e:
                            rospy.loginfo("Failed to convert image %s", str(e))
                    
                    else: # meaning that tracker all fail --> Call Detect
                        detections = self._detector.detect(cv_img, self.conf_th)
                        boxes, clss, confs = extract_yolo_detections(cv_img, detections, self._class_names)
                    
                        objects_detected_dict = dict()
                        objects_detected_dict = self.postprocess(clss, boxes, confs)  # [0_1:{1, [x, y, h, w], confs}, 1_0:{0, x, y, h, w, confs}
                        objects_list = list(objects_detected_dict.keys())  # [0_1, 1_0, ...]
                        print('Tracking the following objects', objects_list)
                        
                        # Recover the Multi-trackers
                        multiple_trackers = dict()
                        if len(objects_list) > 0: # If detect success
                            multiple_trackers = {key: cv2.TrackerKCF_create() for key in objects_list}
                            for item in objects_list:
                                multiple_trackers[item].init(cv_img, objects_detected_dict[item][1])
                        else:
                            print("[Loop stage] Recover thru Dections fail --> Back to Initialization stage")
                            status = 0
            else:
                if rgb_msg is None:
                    print("rgb msg is NONE")
                elif depth_msg is None:
                    print("depth msg is NONE")
                else:
                    print("cam info is NONE")


  

def main():
    rospy.init_node("test_node")
    node = ObjectLocatorNode()
    print("[DEBUG] Node Initialization Done ~")
    node.run()

if __name__ == '__main__':
    main()
