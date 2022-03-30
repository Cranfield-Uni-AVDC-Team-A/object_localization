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
    //nh_private_.param("rgb_image_topic", rgbTopic_, std::string("/zed2/zed_node/left/image_rect_color"));
    // nh_private_.param("depth_image_topic", depthTopic_, std::string("/zed2/zed_node/depth/depth_registered"));
    nh_private_.param("object_detections_topic", objectDetectionsTopic_, std::string("/situational_awareness/object_detections"));
    nh_private_.param("overlay_image_topic", overlayImageTopic_, std::string("/situational_awareness/overlay_image"));
    nh_private_.param("publish_overlay", publish_overlay_, false);
    nh_private_.param("timer_duration", duration_, 0.1);

    /* Setup the Synchronized subscriber */
    //rgbSubscriber_.subscribe(imageTransport_, rgbTopic_, 3);
    //depthSubscriber_.subscribe(imageTransport_, depthTopic_, 3);
    //sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>(5), rgbSubscriber_, depthSubscriber_));
    //sync_->registerCallback(boost::bind(&ObjectLocatorNode::cameraCallback, this, _1, _2));

    /* Setup the Timer to Run Detection every 0.1s */
    timer_ = nh_.createWallTimer(ros::WallDuration(duration_), &ObjectLocatorNode::timerCallback, this);


    /**************************************************
    Setup the ZED Camera directly through the SDK API 
    ***************************************************/

    // Opening the ZED camera before the model deserialization to avoid cuda context issue
    init_parameters_.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters_.sdk_verbose = true;
    init_parameters_.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    //init_parameters_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD; // Used in ROS (REP 103)

    // Open the camera
    auto returned_state = zed_.open(init_parameters_);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
    }
    zed_.enablePositionalTracking();

    // Custom Object Detection
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_mask_output = false; // designed to give person pixel mask
    detection_parameters.detection_model = sl::DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed_.enableObjectDetection(detection_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed_.close();
    }

    // Camera Configuration
    auto camera_config = zed_.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed_.getCameraInformation(pc_resolution).camera_configuration;
    cam_w_pose.pose_data.setIdentity();

    /* Setup the Publisher */
    objectDetectionsPublisher_ = nh_.advertise<uav_stack_msgs::Detector3DArray>(objectDetectionsTopic_,10, false);
    counter_ = 1;
    if(publish_overlay_ == true) {
        overlayImagePublisher_ = imageTransport_.advertise(overlayImageTopic_, 2);
    }

    // Initialized the YOLO Detector
    detector_.reset(new Detector(ros::package::getPath("object_localization") + "/models/" + engine_name, yoloW, yoloH, yoloClasses, yoloThresh, yoloNms));

} 

ObjectLocatorNode::~ObjectLocatorNode(){}

void ObjectLocatorNode::timerCallback(const ros::WallTimerEvent& event)
{
    if (zed_.grab() == sl::ERROR_CODE::SUCCESS)
    {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        zed_.retrieveImage(left_sl, sl::VIEW::LEFT);

        // Preparing inference
        cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
        cv::cvtColor(left_cv_rgba, left_cv_rgb, cv::COLOR_BGRA2BGR); //remove alpha channel from RGB or BGR image
        if (left_cv_rgb.empty()) {std::cout << "Image is empty ! Error !" << std::endl;}

        // Detect

        std::vector<sl::CustomBoxObjectData> detections2D_array = detector_->detect(left_cv_rgb);


        // Send the custom detected boxes to the ZED SDK
        zed_.ingestCustomBoxObjects(detections2D_array);

        // Retrieve the tracked objects, with 2D and 3D attributes
        zed_.retrieveObjects(objects, objectTracker_parameters_rt);
        //zed_.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
        //zed_.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);

	std_msgs::Header header;
	header.seq = counter_;
	header.stamp = ros::Time::now();
	header.frame_id = "left_camera_optical_frame";
        uav_stack_msgs::Detector3DArray detection_message = composeMessages(objects, header);
        objectDetectionsPublisher_.publish(detection_message); // Publish it ~!

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double time_spent = (double) std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
	double FPS = 1.0 / (time_spent * 0.000001);
	std::cout << "time spent= " << time_spent << " [us]" << std::endl;
	std::cout << "FPS = " << FPS << std::endl;

	// Draw the Overlay Image
        if(publish_overlay_ == true)
	{   
            drawDetections(left_cv_rgb, detections2D_array);
            sensor_msgs::ImagePtr overlay_img_msg = cv_bridge::CvImage(header, "bgr8", left_cv_rgb).toImageMsg();
            overlayImagePublisher_.publish(overlay_img_msg);
        }
	counter_++;
    }
}


uav_stack_msgs::Detector3DArray ObjectLocatorNode::composeMessages(sl::Objects objects, std_msgs::Header current_header)
{   
    uav_stack_msgs::Detector3DArray detection3D_array;
    detection3D_array.header = current_header;

    for(auto object : objects.object_list){
        uav_stack_msgs::Detector3D detection3D;
        detection3D.header = current_header;
	
	// ObjectHypothesis
        detection3D.results.unique_id = object.unique_object_id;   // string
        detection3D.results.class_id = object.raw_label;           // string
        detection3D.results.score = object.confidence;             // float
	
	// Tracking
	detection3D.tracker_id = object.id;

	sl::OBJECT_TRACKING_STATE object_tracking_state = object.tracking_state;
	if(object_tracking_state == sl::OBJECT_TRACKING_STATE::OK) 
		{ detection3D.is_tracked = true; }
	else 
		{ detection3D.is_tracked = false; }
        
	// Bounding box
        detection3D.bbox.size_x = (float) abs((object.bounding_box_2d[1][0] - object.bounding_box_2d[0][0]) / 2.0);
        detection3D.bbox.size_y = (float) abs((object.bounding_box_2d[3][1] - object.bounding_box_2d[0][1]) / 2.0);
        
        detection3D.bbox.center.x = (double) (object.bounding_box_2d[0][0] + (detection3D.bbox.size_x / 2));
        detection3D.bbox.center.y = (double) (object.bounding_box_2d[0][1] + (detection3D.bbox.size_y / 2));
	
	// Position w.r.t Camera Frame
        detection3D.position.x = (double) object.position[0];
        detection3D.position.y = (double) object.position[1];
        detection3D.position.z = (double) object.position[2];
	
	// Velocity w.r.t Camera Frame
        detection3D.velocity.x = (double) object.velocity[0];
        detection3D.velocity.y = (double) object.velocity[1];
        detection3D.velocity.z = (double) object.velocity[2];
        
	detection3D_array.detections.push_back(detection3D);
    }

    return detection3D_array;

}

void ObjectLocatorNode::drawDetections(cv::Mat &rgb_image, std::vector<sl::CustomBoxObjectData> &detections2D_array)
{
    for(auto &detection : detections2D_array){
        cv::Point pt1(detection.bounding_box_2d[0][0], detection.bounding_box_2d[0][1]);
        cv::Point pt2(detection.bounding_box_2d[2][0], detection.bounding_box_2d[2][1]);

        cv::rectangle(rgb_image, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::putText(rgb_image, std::to_string(detection.label), cv::Point(pt1.x, pt1.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

void ObjectLocatorNode::print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}


// Mapping between MAT_TYPE and CV_TYPE
int ObjectLocatorNode::getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
    	case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    	case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    	case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    	case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    	case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    	case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    	case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    	case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

cv::Mat ObjectLocatorNode::slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}













}
