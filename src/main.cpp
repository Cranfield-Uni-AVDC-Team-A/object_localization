#include "object_localization/ObjectLocatorNode.h"
#include <chrono>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_locator_ros");

    situational_awareness::ObjectLocatorNode node;

    ros::spin();
    return 0;
    
}
