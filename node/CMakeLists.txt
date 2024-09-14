add_executable(online_node online_node.cpp)
target_link_libraries(online_node super_vio ${ONNX_LIBS} ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES} ${g2o_LIBRARIES} logger utilities )

add_executable(offline_node offline_node.cpp)
target_link_libraries(offline_node super_vio ${ONNX_LIBS} ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${g2o_LIBRARIES} logger utilities )

add_executable(offline_kitti_node offline_kitti_node.cpp)
target_link_libraries(offline_kitti_node super_vio ${ONNX_LIBS} ${OpenCV_LIBS} ${Sophus_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${g2o_LIBRARIES} logger utilities )
