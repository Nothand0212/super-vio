# third party include path
include_directories(
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/thirdparty/sophus
    )


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("EIGEN3_VERSION: " ${EIGEN3_VERSION})
message("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})


set(OpenCV_DIR /home/lin/Projects/opencv-4.2.0/install)
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)
set(OpenCV_LIB_DIR ${OpenCV_DIR}/lib)
file(GLOB OpenCV_LIBS "${OpenCV_LIB_DIR}/*.so")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# # OpenCV
# set(OpenCV_CUDA_MODULE_PATH "")
# find_package(OpenCV REQUIRED QUIET)
# message("OpenCV version: " ${OpenCV_VERSION})
# include_directories(${OpenCV_INCLUDE_DIRS})


# PCL
find_package(PCL REQUIRED)
message("PCL version: " ${PCL_VERSION})
include_directories(${PCL_INCLUDE_DIRS})

# g2o
set(g2o_LIBRARIES
    /usr/local/lib/libg2o_core.so
    /usr/local/lib/libg2o_stuff.so
    /usr/local/lib/libg2o_solver_csparse.so
    /usr/local/lib/libg2o_csparse_extension.so
)
include_directories(${g2o_INCLUDE_DIRS})

