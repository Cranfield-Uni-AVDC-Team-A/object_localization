/*
 * ObjectLocator.hpp
 *
 *  Created on: March 22, 2022
 *      Author: Tiga Ho Yin Leung
 *   Institute: Cranfield University, United Kingdom
 */

#pragma once

// c++
#include <algorithm>
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <semaphore.h>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

// uav_stack_msgs
#include <uav_stack_msgs/BoundingBox2D.h>
#include <uav_stack_msgs/Detector2D.h>
#include <uav_stack_msgs/Detector2DArray.h>
#include <uav_stack_msgs/ObjectHypothesis.h>

//! Bounding box of the detected object.
typedef struct
{
  float x, y, w, h, prob, z;
  int num, Class;
} RosBox_;

class ObjectLocator
{
public:
  /*!
   * Constructor.
   */
  explicit ObjectLocator(ros::NodeHandle nh);

  /*!
   * Destructor.
   */
  ~ObjectLocator();

private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Initialize the ROS connections.
   */
  void init();

  /*!
   * Callback of camera.
   * @param[in] dmap_msg depth map pointer.
   */
  void zedCameraCallback(const sensor_msgs::ImageConstPtr &img_msg,
                         const sensor_msgs::ImageConstPtr &dmap_msg);

  /*!
   * Callback of bounding boxes array.
   * @param[in] dmap_msg depth map pointer.
   */
  void boundingBoxesCallback(const uav_stack_msgs::Detector2DArrayConstPtr &dmap_msg);

  //! ROS node handle.
  ros::NodeHandle nodeHandle_;

  //! Class labels.
  int numClasses_;
  std::vector<std::string> classLabels_;

  //! Advertise and subscribe to image topics.
  image_transport::ImageTransport imageTransport_;

  //! ROS subscriber and publisher.
  image_transport::SubscriberFilter imageSubscriber_; // rgb image (on that detection time)
  image_transport::SubscriberFilter dmapSubscriber_;  // depth map
  ros::Subscriber boundingBoxesSubscriber_;
  ros::Publisher objectInformationPublisher_;

  // Topic synchronization
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image>
      ApproxTimePolicy;
  message_filters::Synchronizer<ApproxTimePolicy> imgSync_;

  //! Detected objects.
  std::vector<std::vector<RosBox_>> rosBoxes_;
  std::vector<int> rosBoxCounter_;

  //! 3D Objects information
  RosBox_ *roiBoxes_;
  uav_stack_msgs::Detector3DArray objectionInformationArray_;

  //! Camera related parameters.
  int frameWidth_;
  int frameHeight_;
  bool zed;
  bool enable_depth_;

  std_msgs::Header imageHeader_;
  cv::Mat camImageCopy_;
  cv::Mat camDmapCopy_;
  boost::shared_mutex mutexImageCallback_;

  bool imageStatus_ = false;
  boost::shared_mutex mutexImageStatus_;

  bool isNodeRunning_ = true;
  boost::shared_mutex mutexNodeStatus_;

  float getObjDepth(float xmin, float xmax, float ymin, float ymax);

  bool getImageStatus(void);

  bool isNodeRunning(void);
}