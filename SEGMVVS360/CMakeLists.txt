cmake_minimum_required(VERSION 3.0.2)
project(SEGMVVS360)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  visp_bridge
  message_filters
  sensor_msgs
  cv_bridge
  message_generation
)

find_package(VISP 3 REQUIRED)

add_service_files(
  FILES
    building_segmentation.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
 CATKIN_DEPENDS rospy message_runtime std_msgs sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)

set(SOURCES nodes/SEGMVVS360.cpp)

add_executable(SEGMVVS360 ${SOURCES})
target_link_libraries(SEGMVVS360 ${catkin_LIBRARIES} ${VISP_LIBRARIES})
add_dependencies(SEGMVVS360 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  nodes/real_image_publisher.py
  nodes/building_segmentation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)