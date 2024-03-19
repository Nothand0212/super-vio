cmake_minimum_required(VERSION 3.16) # 他妈的问题出在我只有/usr/share/cmake-3.16/Modules/FindCUDA.cmake，只能指定3.16.其他都错
project(super_vio  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-DRELEASE)
endif()

include_directories(
/usr/local/include    
/usr/local/include/eigen3
)

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
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

# SPDLOG
find_package(spdlog REQUIRED)
message("spdlog version: "${spdlog_VERSION})

# OEPN CV
set(OpenCV_CUDA_MODULE_PATH "")
find_package(OpenCV REQUIRED QUIET)
message("OpenCV version: " ${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
message("PCL version: " ${PCL_VERSION})
include_directories(${PCL_INCLUDE_DIRS})

# fmt
# find_package(fmt REQUIRED)
# set (FMT_LIBRARIES fmt::fmt)

# SOPHUS
find_package(Sophus REQUIRED)
message("Sophus version: " ${Sophus_VERSION})

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
    include
    ${Sophus_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    )

# logger lib
add_library(logger src/logger/logger.cpp)
target_link_libraries(logger spdlog::spdlog)

add_library(image_process src/image_process.cpp)
target_link_libraries(image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(extractor src/extractor/extractor.cpp)
target_link_libraries(extractor onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(matcher src/matcher/matcher.cpp)
target_link_libraries(matcher onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(frame src/frame.cpp)
target_link_libraries(frame ${OpenCV_LIBS} ${Sophus_LIBS} logger spdlog::spdlog)

add_library(map_point src/data/mapPoint.cpp)
target_link_libraries(map_point ${OpenCV_LIBS} ${Sophus_LIBS} logger spdlog::spdlog)

add_executable(online_node node/online_node.cpp)
target_link_libraries(online_node extractor matcher frame map_point ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES} logger spdlog::spdlog )

add_executable(offline_node node/offline_node.cpp)
target_link_libraries(offline_node extractor matcher frame map_point ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} logger spdlog::spdlog )

add_executable(offline_kitti_node node/offline_kitti_node.cpp)
target_link_libraries(offline_kitti_node extractor matcher frame map_point ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} logger spdlog::spdlog )
