cmake_minimum_required(VERSION 3.16) # 他妈的问题出在我只有/usr/share/cmake-3.16/Modules/FindCUDA.cmake，只能指定3.16.其他都错
project(super_vio  LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS} -lpthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb ${CMAKE_CXX_FLAGS}")

set(CMAKE_INSTALL_RPATH "/home/lin/Projects/third_part/OpenCV/opencv-4.2.0/install/lib")


include(cmake/packages.cmake)

# ROS
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    pcl_conversions
    pcl_ros
    )

# CUDA
find_package(CUDA REQUIRED)

# ONNXRUNTIME
set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-gpu-1.16.3)
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
file(GLOB ONNX_LIBS "${ONNXRUNTIME_ROOTDIR}/lib/*.so")

catkin_package(
    INCLUDE_DIRS include
    DEPENDS PCL
    CATKIN_DEPENDS 
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    pcl_conversions
    pcl_ros
    )
    
# include
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Sophus_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    )

add_subdirectory(src)
add_subdirectory(node)
