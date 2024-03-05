cmake_minimum_required(VERSION 3.16) # 他妈的问题出在我只有/usr/share/cmake-3.16/Modules/FindCUDA.cmake，只能指定3.16.其他都错
project(super-vio  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-DRELEASE)
endif()

# CUDA
find_package(CUDA REQUIRED)

# ONNXRUNTIME
set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-gpu-1.16.3)
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

# SPDLOG
find_package(spdlog REQUIRED)
message(spdlog_VERSION: ${spdlog_VERSION})

# OEPN CV
set(OpenCV_CUDA_MODULE_PATH "")
find_package(OpenCV REQUIRED QUIET)
message(" " ${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

# fmt
find_package(fmt REQUIRED)
message("fmt version: " ${fmt_VERSION})

# SOPHUS
find_package(Sophus REQUIRED)
message("Sophus version: " ${Sophus_VERSION})

# include
include_directories(
    include
    /usr/local/include/eigen3
    ${Sophus_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    
    )

# logger lib
add_library(logger src/logger/logger.cpp)
target_link_libraries(logger spdlog::spdlog)

add_library(image_process src/image_process.cpp)
target_link_libraries(image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(combine_onnx_runner src/combined_onnx_runner.cpp)
target_link_libraries(combine_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(decoupled_onnx_runner src/decoupled_onnx_runner.cpp)
target_link_libraries(decoupled_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(extractor src/extractor/extractor.cpp)
target_link_libraries(extractor onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(matcher src/matcher/matcher.cpp)
target_link_libraries(matcher onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(frame src/frame.cpp)
target_link_libraries(frame ${OpenCV_LIBS} ${Sophus_LIBS} logger spdlog::spdlog)

add_library(map_point src/data/mapPoint.cpp)
target_link_libraries(map_point ${OpenCV_LIBS} ${Sophus_LIBS} logger spdlog::spdlog)


add_executable(cammera_runner test/cammera_runner.cpp)
target_link_libraries(cammera_runner extractor matcher frame map_point ${OpenCV_LIBS} ${Sophus_LIBS} logger spdlog::spdlog fmt::fmt)
