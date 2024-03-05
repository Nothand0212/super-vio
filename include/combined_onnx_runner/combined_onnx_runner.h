/**
 ******************************************************************************
 * @file           : include/combined_onnx_runner.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "configuration.h"
#include "image_process.h"

class CombinedOnnxRunner : public BaseOnnxRunner
{
private:
  // member variables
  unsigned int threads_num_;

  Ort::Env                         env_;
  Ort::SessionOptions              session_options_;
  std::unique_ptr<Ort::Session>    session_uptr_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<char*>                input_node_names_;
  std::vector<std::vector<int64_t>> input_node_shapes_;

  std::vector<char*>                output_node_names_;
  std::vector<std::vector<int64_t>> output_node_shapes_;

  float     match_threshold_{ 0.0f };
  long long timer_{ 0 };

  std::vector<float> scales_{ 1.0f, 1.0f };

  std::vector<Ort::Value> output_tensors_;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_result_;
  std::vector<cv::Point2f>                                      key_points_src_;
  std::vector<cv::Point2f>                                      key_points_dst_;


private:
  cv::Mat preProcess( const Config& config, const cv::Mat& image, float& scale );

  int inference( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst );

  int postProcess( const Config& config );

public:
  explicit CombinedOnnxRunner( unsigned int threads_num = 1 );
  ~CombinedOnnxRunner();

  int initOrtEnv( const Config& config );

  float getMatchThreshold() const;

  void setMatchThreshold( float match_threshold );

  double getTimer( const std::string& name ) const;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeyPointsResult() const;
  std::vector<cv::Point2f>                                      getKeyPointsSrc() const;
  std::vector<cv::Point2f>                                      getKeyPointsDst() const;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> inferenceImagePair( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst );
};
