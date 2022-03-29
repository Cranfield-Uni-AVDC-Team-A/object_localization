#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <uav_stack_msgs/BoundingBox2D.h>
#include <uav_stack_msgs/Detector2D.h>
#include <uav_stack_msgs/Detector2DArray.h>
#include <uav_stack_msgs/Detector3D.h>
#include <uav_stack_msgs/Detector3DArray.h>

#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>

#include "object_localization/Detector.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include <mutex>
#include <condition_variable>



namespace situational_awareness{


class ObjectLocatorNode{

public:
    ObjectLocatorNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    ObjectLocatorNode() : ObjectLocatorNode(ros::NodeHandle(), ros::NodeHandle("~")) {}
    ~ObjectLocatorNode();

    void cameraCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg);
    struct Positions3D{float x; float y; float z; };

private:
    std::vector<ObjectLocatorNode::Positions3D> retrievePosition(std::vector<Detections2D>& detections2D_array,
                                                                 cv_bridge::CvImagePtr depth_image_ptr);
    
    uav_stack_msgs::Detector3DArray composeMessages(std::vector<Detections2D>& detections2D_array, 
                                                        std::vector<ObjectLocatorNode::Positions3D>& positions3D_array,
                                                        std_msgs::Header current_header);
    
    void drawDetections(cv_bridge::CvImagePtr rgb_image_ptr, std::vector<Detections2D>&detections2D_array);
    

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport imageTransport_;
    image_transport::SubscriberFilter rgbSubscriber_;
    image_transport::SubscriberFilter depthSubscriber_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>> sync_;

    ros::Publisher objectDetectionsPublisher_;
    image_transport::Publisher overlayImagePublisher_;

    std::unique_ptr<Detector> detector_;
    //std::vector<uav_stack_msgs::BoundingBox2D> detect_results_;

    std::string rgbTopic_;
    std::string depthTopic_;
    std::string objectDetectionsTopic_;
    std::string overlayImageTopic_;
    
    bool publish_overlay_;


    







};

}
