cmake_minimum_required(VERSION 2.8.3)
project(gaussian_mixture)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
  std_msgs
)

add_service_files(
    FILES
    ComputeGaussianMixture.srv
    ComputeOrientedGaussianMixture.srv
    ComputePhotometricEllipticalGaussianMixture.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  rospy
  sensor_msgs
  std_msgs
)
