add_library(logger SHARED logger/logger.cpp)
target_link_libraries(logger)

add_library(utilities SHARED 
    utilities/configuration.cpp
    utilities/image_process.cpp
)


add_library(super_vio SHARED
    super_vio/mapPoint.cpp
    super_vio/extractor.cpp
    super_vio/matcher.cpp
    super_vio/frame.cpp
    super_vio/features.cpp
)
target_link_libraries(super_vio ${ONNX_LIBS} ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} logger utilities)
