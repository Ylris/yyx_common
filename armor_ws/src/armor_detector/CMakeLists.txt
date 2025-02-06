cmake_minimum_required(VERSION 3.22)
project(armor_detector)

if(POLICY CMP0130)
  cmake_policy(SET CMP0130 NEW)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmorDetection.msg"
  DEPENDENCIES std_msgs
)

add_executable(yolo_detector src/yolo_detector.cpp)

target_include_directories(yolo_detector PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>"
  "$<INSTALL_INTERFACE:include>"
)

ament_target_dependencies(yolo_detector
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  OpenCV
  rosidl_default_runtime  
  rosidl_typesupport_cpp  
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(yolo_detector "${cpp_typesupport_target}")

install(TARGETS yolo_detector
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  std_msgs
  sensor_msgs
)

ament_package()

