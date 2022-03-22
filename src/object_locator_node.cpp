/*
 * object_detector_node.cpp
 *
 *  Created on: March 22, 2022
 *      Author: Tiga Ho Yin Leung
 *   Institute: Cranfield University, United Kingdom
 */

#include <object_localization/ObjectLocator.hpp>
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_localization");
  ros::NodeHandle nodeHandle("~");
  object_localization::ObjectLocator obj_locator(nodeHandle);

  ros::spin();
  return 0;
}
