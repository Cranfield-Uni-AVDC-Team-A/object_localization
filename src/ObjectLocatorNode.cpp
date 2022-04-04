#include "object_localization/ObjectLocatorNode.h"
#include <chrono>

namespace situational_awareness
{

ObjectLocatorNode::ObjectLocatorNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private), imageTransport_(nh)
{
    /* Parameters read from the config file*/
    // YOLO Parameters
    std::string engine_name;
    int yoloW;
    int yoloH;
    int yoloClasses;
    double yoloThresh;
    double yoloNms;
    nh_private_.param("engine_name", engine_name, std::string("yolov5s.engine"));
    nh_private_.param("yolo_width", yoloW, 640);
    nh_private_.param("yolo_height", yoloH, 640);
    nh_private_.param("yolo_classes", yoloClasses, 80);
    nh_private_.param("yolo_detection_threshold", yoloThresh, 0.998);
    nh_private_.param("yolo_nms_threshold", yoloNms, 0.25);

    // Publisher and Subscriber Topics
    nh_private_.param("rgb_image_topic", rgbTopic_, std::string("/zed2/zed_node/left/image_rect_color"));
    nh_private_.param("depth_image_topic", depthTopic_, std::string("/zed2/zed_node/depth/depth_registered"));
    nh_private_.param("object_detections_topic", objectDetectionsTopic_, std::string("/situational_awareness/object_detections"));
    nh_private_.param("overlay_image_topic", overlayImageTopic_, std::string("/situational_awareness/overlay_image"));
    nh_private_.param("publish_overlay", publish_overlay_, false);

    /* Setup the Synchronized subscriber */
    rgbSubscriber_.subscribe(imageTransport_, rgbTopic_, 3);
    depthSubscriber_.subscribe(imageTransport_, depthTopic_, 3);
    sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>(5), rgbSubscriber_, depthSubscriber_));
    sync_->registerCallback(boost::bind(&ObjectLocatorNode::cameraCallback, this, _1, _2));

    /* Setup the Publisher */
    objectDetectionsPublisher_ = nh_.advertise<uav_stack_msgs::Detector3DArray>(objectDetectionsTopic_,2, false);
    
    if(publish_overlay_ == true) {
        overlayImagePublisher_ = imageTransport_.advertise(overlayImageTopic_, 2);
    }

    // Initialized the YOLO Detector
    detector_.reset(new Detector(ros::package::getPath("object_localization") + "/models/" + engine_name, yoloW, yoloH, yoloClasses, yoloThresh, yoloNms));

} 

ObjectLocatorNode::~ObjectLocatorNode(){}

void ObjectLocatorNode::cameraCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg)
{
    cv_bridge::CvImagePtr rgb_image_ptr;
    cv_bridge::CvImagePtr depth_image_ptr;
    
    // Convert ROS Image MSG to CV::MAT
    try {
        rgb_image_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        depth_image_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Detect
    std::vector<Detections2D> detections2D_array = detector_->detect(rgb_image_ptr);

    // Locate
    std::vector<ObjectLocatorNode::Positions3D> positions3D_array
        = retrievePosition(detections2D_array, depth_image_ptr);

    // Compose the message
    uav_stack_msgs::Detector3DArray detection_message = composeMessages(detections2D_array, 
                                                                        positions3D_array, 
                                                                        rgb_msg->header);
    objectDetectionsPublisher_.publish(detection_message); // Publish it ~!

    
    if(publish_overlay_ == true){   
        // Draw the Overlay Image
        drawDetections(rgb_image_ptr, detections2D_array);
        overlayImagePublisher_.publish(rgb_image_ptr->toImageMsg());
    }
}

uav_stack_msgs::Detector3DArray ObjectLocatorNode::composeMessages(std::vector<Detections2D>& detections2D_array, 
                                                        std::vector<ObjectLocatorNode::Positions3D>& positions3D_array,
                                                        std_msgs::Header current_header)
{   
    uav_stack_msgs::Detector3DArray detection3D_array;
    detection3D_array.header = current_header;

    for (size_t i = 0; i<detections2D_array.size(); i++){
        
        //Detections2D detection2D = detections2D_array[i];
        //ObjectLocatorNode::Positions3D pos_3D = positions3D_array[i];
        uav_stack_msgs::Detector3D detection3D;

        detection3D.header = current_header;
        
        detection3D.results.id = std::to_string(detections2D_array[i].classID) + "_" + std::to_string(current_header.stamp.sec);
        detection3D.results.score = detections2D_array[i].prob;

        detection3D.bbox.center.x = (double) detections2D_array[i].rectangle_box.x + 
                                            (detections2D_array[i].rectangle_box.width/2);

        detection3D.bbox.center.y = (double) detections2D_array[i].rectangle_box.y + 
                                            (detections2D_array[i].rectangle_box.height/2);

        detection3D.bbox.size_x = (float) detections2D_array[i].rectangle_box.width;
        detection3D.bbox.size_y = (float) detections2D_array[i].rectangle_box.height;

        detection3D.position.x = (double) positions3D_array[i].x;
        detection3D.position.y = (double) positions3D_array[i].y;
        detection3D.position.z = (double) positions3D_array[i].z;

        detection3D_array.detections.push_back(detection3D);
    }

    return detection3D_array;

}



std::vector<ObjectLocatorNode::Positions3D> ObjectLocatorNode::retrievePosition(std::vector<Detections2D>& detections2D_array, cv_bridge::CvImagePtr depth_image_ptr)
{
    cv::Mat depth_mat = depth_image_ptr->image; 
    std::vector<ObjectLocatorNode::Positions3D> positions3D_array;
    float cam_info_cx = 619.28302;
    float cam_info_cy = 58.944061;
    float cam_info_fx = 529.12689;
    float cam_info_fy = 529.12689;

    for(auto &detection : detections2D_array){
        //float center_x = detection.rectangle_box.x + (detection.rectangle_box.width/2);
        //float center_y = detection.rectangle_box.y + (detection.rectangle_box.height/2);
        float xmin = detection.rectangle_box.x;
        float ymin = detection.rectangle_box.y;
        float xmax = detection.rectangle_box.x + detection.rectangle_box.width;
        float ymax = detection.rectangle_box.y + detection.rectangle_box.height;
        std::vector<float>depth_arr;
        float x, y, d;
        int refs = 3;
        float depth;
        for (int i = 1; i < refs+1; ++i)
        {
            for(int j = 1; j < refs+1; ++j)
            {
                x = xmin + j*(xmax-xmin)/(refs+1);
                y = ymin + i*(ymax-ymin)/(refs+1);
	            //std::cout << "[DEBUG] X : " << x << " Y: " << y << std::endl;
                d = depth_mat.at<float>((int)center_y, (int)center_x);
                if (std::isnormal(d)) depth_arr.push_back(d);
            }
        }
        std::sort(depth_arr.begin(), depth_arr.end());

        if (depth_arr.size() > 1)
            depth = depth_arr[1];
        else if (depth_arr.size() == 1)
            depth = depth_arr[0];
        else
            depth = NAN;
        
        ObjectLocatorNode::Positions3D pos_3D;
        if (std::isfinite(depth))
        {
            pos_3D.x = depth * ((center_x - cam_info_cx) / cam_info_fx);
            pos_3D.y = depth* ((center_y - cam_info_cy) / cam_info_fy);
            pos_3D.z = depth;

            positions3D_array.push_back(pos_3D);
        }
        else {
            std::cout << "The Distance can not be computed" << std::endl;
            pos_3D.x = 0.0;
            pos_3D.y = 0.0;
            pos_3D.z  = 0.0;
        }
    }
    return positions3D_array;
}



void ObjectLocatorNode::drawDetections(cv_bridge::CvImagePtr rgb_image_ptr, std::vector<Detections2D> &detections2D_array)
{
    cv::Mat img = rgb_image_ptr->image;
    for(auto &detection : detections2D_array){
        cv::rectangle(img, detection.rectangle_box, cv::Scalar(0, 255, 0));
        cv::putText(img, std::to_string((int) detection.classID), cv::Point(detection.rectangle_box.x, detection.rectangle_box.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}
















}
