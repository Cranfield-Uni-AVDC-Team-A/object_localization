cmake_minimum_required(VERSION 3.0.2)
project(object_localization)

####################
##     ZED SDK    ##
#####################

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0004 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(SPECIAL_OS_LIBS "")

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  uav_stack_msgs
  vision_msgs
)  

catkin_package(
  INCLUDE_DIRS 
    include
  LIBRARIES 
    object_localization
  CATKIN_DEPENDS 
  cv_bridge 
  image_transport 
  message_filters
  roscpp 
  rospy 
  sensor_msgs 
  std_msgs 
  uav_stack_msgs 
  vision_msgs
  #DEPENDS system_lib
)
############################################################
##        Include path for our own headers/library        ##
############################################################
include_directories(${catkin_INCLUDE_DIRS})

############################################################
##        Include path for Required 3rd party library     ##
############################################################

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})


link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


#----------------------------#
#     Cuda and TensorRT      #
#----------------------------#
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HDR_FILES include/*.h*)
FILE(GLOB_RECURSE PACKAGE_HDR_FILES include/object_localization/*.h*)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
cuda_add_executable(${PROJECT_NAME} ${HDR_FILES} ${PACKAGE_HDR_FILES} ${SRC_FILES})
add_definitions(-std=c++14 -g -O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)


#----------------#
#     ZED SDK    #
#----------------#

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

SET(TRT_LIBS nvinfer)


########################################################
##        Building links to the 3rd party library     ##
########################################################

target_link_libraries(${PROJECT_NAME}
                        ${TRT_LIBS}
                        ${SPECIAL_OS_LIBS}
                        ${ZED_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

#############
## Install ##
#############

catkin_install_python(PROGRAMS
   sripts/test_node.py
   sripts/test_sub.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

