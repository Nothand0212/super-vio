/**
 ******************************************************************************
 * @file           : configuration.hpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"
#include "string"
#include "utilities/color.hpp"

//Config
namespace utilities
{
class Configuration
{
public:
  // **** camera related ****
  std::string camera_json_path              = "/home/lin/Projects/super-vio/config/camera.json";
  int         camera_port                   = 0;
  int         camera_rate                   = 30;
  int         camera_width                  = 640;
  int         camera_height                 = 480;
  cv::Mat     distortion_coefficients_left  = cv::Mat::zeros( 1, 5, CV_32FC1 );
  cv::Mat     distortion_coefficients_right = cv::Mat::zeros( 1, 5, CV_32FC1 );
  cv::Mat     camera_matrix_left            = cv::Mat::eye( 3, 3, CV_32FC1 );
  cv::Mat     camera_matrix_right           = cv::Mat::eye( 3, 3, CV_32FC1 );
  // **** camera related ****

  std::string kitti_path     = "/home/lin/Projects/super-vio/data/kitti";
  std::string matcher_path   = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_lightglue_fused.onnx";  // light_glue
  std::string extractor_path = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint.onnx";                  // only super point
  std::string combiner_path  = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_2048_lightglue_end2end.onnx";

  std::string image_src_path = "/home/lin/Projects/vision_ws/data/left";
  std::string image_dst_path = "/home/lin/Projects/vision_ws/data/right";

  std::string output_path = "/home/lin/CLionProjects/light_glue_onnx/output";
  std::string log_path    = "/home/lin/Projects/super-vio/log_data/tmp.log";

  bool gray_flag = true;

  unsigned int image_size = 2048;
  float        threshold  = 0.05f;

  std::string device{ "cuda" };

  // **** FrontEnd Related ****
  bool  enable_backend_{ false };
  bool  enable_imu_preintegration_{ false };
  float extractor_score_threshold_{ 0.5 };

public:
  Configuration()  = default;
  ~Configuration() = default;
  void readConfigFile( const std::string& config_file_path );
};
}  // namespace utilities