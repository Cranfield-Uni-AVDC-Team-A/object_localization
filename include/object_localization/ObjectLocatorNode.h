#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cvconfig.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include <mutex>
#include <condition_variable>

/* ZED SDK */
#include <sl/Camera.hpp>



namespace situational_awareness
{


class ObjectLocatorNode
{

public:
    ObjectLocatorNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    ObjectLocatorNode() : ObjectLocatorNode(ros::NodeHandle(), ros::NodeHandle("~")) {}
    ~ObjectLocatorNode();

    void cameraCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg);
    struct Positions3D
    {
        float x;
        float y;
        float z;
    };
    void timerCallback(const ros::WallTimerEvent& event);

private:
    std::vector<ObjectLocatorNode::Positions3D> retrievePosition(std::vector<Detections2D>& detections2D_array,
            cv_bridge::CvImagePtr depth_image_ptr);

    geometry_msgs::Point retrievePositionFromDepth(sl::ObjectData &object, sl::Mat &depth_mat);

    uav_stack_msgs::Detector3DArray composeMessages(sl::Objects &objects, std_msgs::Header current_header, sl::Resolution img_resolution);

    void drawDetections(cv::Mat &rgb_image, std::vector<sl::CustomBoxObjectData> &detections2D_array);

    void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix);

    cv::Mat slMat2cvMat(sl::Mat& input);
    int getOCVtype(sl::MAT_TYPE type);

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport imageTransport_;
    //image_transport::SubscriberFilter rgbSubscriber_;
    //image_transport::SubscriberFilter depthSubscriber_;
    //std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>> sync_;

    ros::Publisher objectDetectionsPublisher_;
    image_transport::Publisher overlayImagePublisher_;

    ros::WallTimer timer_;

    uint32_t counter_;

    std::unique_ptr<Detector> detector_;
    //std::vector<uav_stack_msgs::BoundingBox2D> detect_results_;

    std::string rgbTopic_;
    std::string depthTopic_;
    std::string objectDetectionsTopic_;
    std::string overlayImageTopic_;

    bool publish_overlay_;
    double duration_;
    bool sdk_locating_;

    /* ZED SDK Instance */
    sl::Camera zed_;
    sl::InitParameters init_parameters_;

    sl::Mat left_sl, point_cloud;
    cv::Mat left_cv_rgb;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;

    float cam_info_cx = 619.2830200195312;
    float cam_info_cy = 58.9440612792969;
    float cam_info_fx = 529.1268920898438;
    float cam_info_fy = 529.1268920898438;










};

}
