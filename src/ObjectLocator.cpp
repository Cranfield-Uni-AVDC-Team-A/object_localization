/*
 * ObjectLocator.cpp
 *
 *  Created on: March 22, 2022
 *      Author: Tiga Ho Yin Leung
 *   Institute: Cranfield University, United Kingdom
 */

#include "object_localization/ObjectLocator.hpp"

namespace ObjectLocalization
{
    sem_t sem_new_image_; // Semaphore to indicate new image

    ObjectLocator::ObjectLocator(ros::NodeHandle nh)
        : nodeHandle_(nh),
          imageTransport_(nodeHandle),
          rosBoxes_(0),
          rosBoxCounter_(0),
          imgSync_(ApproxTimePolicy(3), imageSubscriber_, dmapSubscriber_)
    {
        ROS_INFO("[ObjectLocator] Node Started");

        // Read parameters from config file.
        if (!readParameters())
        {
            ros::requestShutdown();
        }

        // Initialize the Object Locator
        init();
    }

    ObjectLocator::~ObjectLocator()
    {
        {
            boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
            isNodeRunning_ = false;
        }
        sem_destroy(&sem_new_image_)
    }

    bool ObjectLocator::readParameters()
    {
        // Load common parameters.
        nodeHandle_.param("locating_method/enable_depth", enable_depth, true);      // Using Depth
        nodeHandle_.param("locating_method/enable_pinhole", enable_pinhole, false); // Using Pin Hole Camera Model Projection

        // Set vector sizes.
        nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                          std::vector<std::string>(0));
        numClasses_ = classLabels_.size();
        rosBoxes_ = std::vector<std::vector<RosBox_>>(numClasses_);
        rosBoxCounter_ = std::vector<int>(numClasses_);

        return true;
    }

    void ObjectLocator::init()
    {
        ROS_INFO("[ObjectLocator] Initializing~");

        // Initialize semaphore
        sem_init(&sem_new_image_, 0, 0);

        // ZED camera
        nodeHandle_.param("zed_enable", zed, true);

        // Initialize publisher and subscriber.
        std::string ns;
        nodeHandle_.getParam("namespace", ns);

        // Sub
        std::string cameraTopicName;
        int cameraQueueSize;
        nodeHandle_.param("subscribers/camera_reading/rgb_topic", cameraTopicName,
                          std::string("/camera/image_raw"));
        nodeHandle_.param("subscribers/camera_reading/rgb_queue_size", cameraQueueSize, 1);

        std::string dmapTopicName;
        int dmapQueueSize;
        nodeHandle_.param("subscribers/camera_reading/dmap_topic", dmapTopicName,
                          std::string("/camera/dmap"));
        nodeHandle_.param("subscribers/camera_reading/dmap_queue_size", dmapQueueSize, 1);

        std::string boundingBoxesTopicName;
        int boundingBoxesImageQueueSize;
        nodeHandle_.param("subscribers/detections_reading/bounding_box_topic", boundingBoxesTopicName,
                          std::string("/detections"));
        nodeHandle_.param("subscribers/detections_reading/bounding_box_queue_size", boundingBoxesImageQueueSize, 1);

        // Pub
        std::string objectInformationTopicName;
        int objectInformationQueueSize;
        bool objectInformationLatch;
        nodeHandle_.param("publishers/object_information/topic", objectInformationTopicName,
                          std::string("/object_3d_information"));
        nodeHandle_.param("publishers/object_information/queue_size", objectInformationQueueSize, 1);
        nodeHandle_.param("publishers/object_information/latch", objectInformationLatch, false);

        if (ns.length() > 0)
        {
            objectInformationTopicName = "/" + ns + objectInformationTopicName;
        }

        imageSubscriber_.subscribe(imageTransport_, cameraTopicName, cameraQueueSize);
        dmapSubscriber_.subscribe(imageTransport_ dmapTopicName, dmapQueueSize);
        imgSync_.connectInput(imageSubscriber_, dmapSubscriber_);
        imgSync_.registerCallback(boost::bind(&ObjectLocator::zedCameraCallback, this, _1, _2));

        boundingBoxesSubscriber_ = nodeHandle_.subscribe(boundingBoxesTopicName, 1, &ObjectLocator::boundingBoxesCallback, this)

                                       objectInformationPublisher_ = nodeHandle_.advertise<uav_stack_msgs::Detector2DArray>(objectInformationTopicName, objectInformationQueueSize, objectInformationLatch);

        ROS_INFO("Waiting for images in topic: %s", imageSubscriber_.getTopic().c_str());
    }

    void ObjectLocator::zedCameraCallback(const sensor_msgs::ImageConstPtr &img_msg,
                                          const sensor_msgs::ImageConstPtr &dmap_msg)
    {
        ROS_DEBUG("[ObjectLocator] Image received.");

        cv_bridge::CvImagePtr cam_image, cam_dmap;

        // Convert ROS Image msg to CV Image
        try
        {
            cam_image = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
            cam_dmap = cv_bridge::toCvCopy(dmap_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (cam_image && cam_dmap)
        {
            {
                boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
                imageHeader_ = img_msg->header;
                //camImageCopy_ = cam_image->image.clone();
                camDmapCopy_ = cam_dmap->image.clone();
            }
            {
                boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
                imageStatus_ = true;
            }
            frameWidth_ = cam_image->image.size().width;
            frameHeight_ = cam_image->image.size().height;
            sem_post(&sem_new_image_);
        }
        return;
    }

    float ObjectLocator::getObjDepth(float xmin, float xmax, float ymin, float ymax)
    {
        /* Given the bounding box, read the depth from 9 internal points. Sort them, then take the second minimum.
         * This is possibly better than taking the minimum as it may be a spurious outlier, for some reason. */
        std::vector<float> depths;
        float x, y, d;
        int refs = 3;

        for (int i = 1; i < refs + 1; ++i)
        {
            for (int j = 1; j < refs + 1; ++j)
            {
                x = xmin + j * (xmax - xmin) / (refs + 1);
                y = ymin + i * (ymax - ymin) / (refs + 1);
                d = camDmapCopy_.at<float>((int)(y * frameHeight_), (int)(x * frameWidth_));
                if (std::isnormal(d))
                    depths.push_back(d);
            }
        }
        std::sort(depths.begin(), depths.end());

        if (depths.size() > 1)
        {
            // printf("depth at (%d, %d): %f\n", (int)((xmin+xmax)/2*frameWidth_), (int)((ymin+ymax)/2*frameHeight_), depths[1]);
            return depths[1];
        }
        else if (depths.size() == 1)
        {
            // printf("depth at (%d, %d): %f\n", (int)((xmin+xmax)/2*frameWidth_), (int)((ymin+ymax)/2*frameHeight_), depths[0]);
            return depths[0];
        }
        else
            return NAN;
    }

    bool ObjectLocator::getImageStatus(void)
    {
        boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
        return imageStatus_;
    }

    bool ObjectLocator::isNodeRunning(void)
    {
        boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
        return isNodeRunning_;
    }

} // namespace ObjectLocalization


/**
 * @todo:
 * Bounding box call back
 * 3d position calculations
 * publish / subscribe in thread
 * add subscriber for camera_infp
 * add tf listener
 */