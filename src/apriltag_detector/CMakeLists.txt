find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ApriltagDetectionArray.msg"
  DEPENDENCIES geometry_msgs 
)
