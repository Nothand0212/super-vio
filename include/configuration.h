/**
 ******************************************************************************
 * @file           : configuration.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once
#include <fstream>

#include "nlohmann/json.hpp"
#include "string"
#include "utilities/color.h"
class Config
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

public:
  Config()  = default;
  ~Config() = default;

  void readConfig( const std::string& config_file_path )
  {
    std::cout << BOLDGREEN << "Read config from: " << config_file_path << RESET << "\n";
    std::ifstream  config_file( config_file_path );
    nlohmann::json config;
    config_file >> config;
    config_file.close();

    matcher_path   = config[ "matcher_path" ];
    extractor_path = config[ "extractor_path" ];
    combiner_path  = config[ "combiner_path" ];

    image_src_path = config[ "image_src_path" ];
    image_dst_path = config[ "image_dst_path" ];

    output_path = config[ "output_path" ];
    log_path    = config[ "log_path" ];


    gray_flag = config[ "gray_flag" ];

    image_size = config[ "image_size" ];
    threshold  = config[ "threshold" ];

    device = config[ "device" ];

    // **** read camera config ****
    std::cout << BOLDGREEN << "Read camera config from: " << camera_json_path << RESET << "\n";
    camera_json_path = config[ "camera_json_path" ];
    std::ifstream  camera_file( camera_json_path );
    nlohmann::json camera_config;
    camera_file >> camera_config;
    camera_file.close();

    camera_port = camera_config[ "port" ];
    std::cout << "camera_port: " << camera_port << "\n";
    camera_rate = camera_config[ "rate" ];
    std::cout << "camera_rate: " << camera_rate << "\n";
    camera_width = camera_config[ "resolution" ][ "width" ];
    std::cout << "camera_width: " << camera_width << "\n";
    camera_height = camera_config[ "resolution" ][ "height" ];
    std::cout << "camera_height: " << camera_height << "\n";


    double f_x = camera_config[ "camera_matrix" ][ "left" ][ "fx" ];
    double f_y = camera_config[ "camera_matrix" ][ "left" ][ "fy" ];
    double c_x = camera_config[ "camera_matrix" ][ "left" ][ "cx" ];
    double c_y = camera_config[ "camera_matrix" ][ "left" ][ "cy" ];

    this->camera_matrix_left.at<float>( 0, 0 ) = f_x;
    this->camera_matrix_left.at<float>( 1, 1 ) = f_y;
    this->camera_matrix_left.at<float>( 0, 2 ) = c_x;
    this->camera_matrix_left.at<float>( 1, 2 ) = c_y;
    std::cout << "camera_matrix_left:\n"
              << this->camera_matrix_left << "\n";

    f_x = camera_config[ "camera_matrix" ][ "right" ][ "fx" ];
    f_y = camera_config[ "camera_matrix" ][ "right" ][ "fy" ];
    c_x = camera_config[ "camera_matrix" ][ "right" ][ "cx" ];
    c_y = camera_config[ "camera_matrix" ][ "right" ][ "cy" ];

    this->camera_matrix_right.at<float>( 0, 0 ) = f_x;
    this->camera_matrix_right.at<float>( 1, 1 ) = f_y;
    this->camera_matrix_right.at<float>( 0, 2 ) = c_x;
    this->camera_matrix_right.at<float>( 1, 2 ) = c_y;
    std::cout << "camera_matrix_right:\n"
              << this->camera_matrix_right << "\n";


    double k1 = camera_config[ "distortion_coefficients" ][ "left" ][ "k1" ];
    double k2 = camera_config[ "distortion_coefficients" ][ "left" ][ "k2" ];
    double p1 = camera_config[ "distortion_coefficients" ][ "left" ][ "p1" ];
    double p2 = camera_config[ "distortion_coefficients" ][ "left" ][ "p2" ];
    double k3 = camera_config[ "distortion_coefficients" ][ "left" ][ "k3" ];

    this->distortion_coefficients_left.at<float>( 0, 0 ) = k1;
    this->distortion_coefficients_left.at<float>( 0, 1 ) = k2;
    this->distortion_coefficients_left.at<float>( 0, 2 ) = p1;
    this->distortion_coefficients_left.at<float>( 0, 3 ) = p2;
    this->distortion_coefficients_left.at<float>( 0, 4 ) = k3;

    k1 = camera_config[ "distortion_coefficients" ][ "right" ][ "k1" ];
    k2 = camera_config[ "distortion_coefficients" ][ "right" ][ "k2" ];
    p1 = camera_config[ "distortion_coefficients" ][ "right" ][ "p1" ];
    p2 = camera_config[ "distortion_coefficients" ][ "right" ][ "p2" ];
    k3 = camera_config[ "distortion_coefficients" ][ "right" ][ "k3" ];

    this->distortion_coefficients_right.at<float>( 0, 0 ) = k1;
    this->distortion_coefficients_right.at<float>( 0, 1 ) = k2;
    this->distortion_coefficients_right.at<float>( 0, 2 ) = p1;
    this->distortion_coefficients_right.at<float>( 0, 3 ) = p2;
    this->distortion_coefficients_right.at<float>( 0, 4 ) = k3;
  }
};