/**
 ******************************************************************************
 * @file           : include/decoupled_onnx_runner/decoupled_onxx_runner.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "configuration.h"
#include "image_process.h"
#include "utilities/accumulate_average.h"
#include "utilities/timer.h"

class DecoupledOnnxRunner : public BaseOnnxRunner
{
private:
  unsigned int threads_num_;

  Ort::Env                      env_extractor_, env_matcher_;
  Ort::SessionOptions           session_options_extractor_, session_options_matcher_;
  std::unique_ptr<Ort::Session> session_uptr_extractor_, session_uptr_matcher_;

  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<char*>                input_node_names_extractor_;
  std::vector<std::vector<int64_t>> input_node_shapes_extractor_;
  std::vector<char*>                output_node_names_extractor_;
  std::vector<std::vector<int64_t>> output_node_shapes_extractor_;


  std::vector<char*>                input_node_names_matcher_;
  std::vector<std::vector<int64_t>> input_node_shapes_matcher_;
  std::vector<char*>                output_node_names_matcher_;
  std::vector<std::vector<int64_t>> output_node_shapes_matcher_;

  float             match_threshold_{ 0.0f };
  Timer             timer_extractor_, timer_matcher_;
  AccumulateAverage average_timer_extractor_, average_timer_matcher_;

  std::vector<float> scales_{ 1.0f, 1.0f };

  std::vector<std::vector<Ort::Value>> output_tensors_extoractor_;  // two dimension for image_src and image_dst
  std::vector<Ort::Value>              output_tensor_matcher_;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_result_;
  std::vector<cv::Point2f>                                      key_points_src_;
  std::vector<cv::Point2f>                                      key_points_dst_;

private:
  cv::Mat                                     preProcessExtractor( const Config& config, const cv::Mat& image, float& scale );
  int                                         inferenceExtractor( const Config& config, const cv::Mat& image );  // run each image separately
  std::pair<std::vector<cv::Point2f>, float*> postProcessExtractor( const Config& config, std::vector<Ort::Value> tensor );


  std::vector<cv::Point2f> preProcessMatcher( std::vector<cv::Point2f> key_points, const int& height, const int& width );
  int                      inferenceMatcher( const Config& config, std::vector<cv::Point2f> key_points_src, std::vector<cv::Point2f> key_points_dst, float* descriptor_src, float* descriptor_dst );
  int                      postProcessMatcher( const Config& config, std::vector<cv::Point2f> key_points_src, std::vector<cv::Point2f> key_points_dst );


public:
  explicit DecoupledOnnxRunner( unsigned int threads_num = 1 );
  ~DecoupledOnnxRunner();

  int initOrtEnv( const Config& config );

  float getMatchThreshold() const;
  void  setMatchThreshold( float threshold );

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeyPointsResult() const;
  std::vector<cv::Point2f>                                      getKeyPointsSrc() const;
  std::vector<cv::Point2f>                                      getKeyPointsDst() const;

  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> inferenceImagePair( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst );
};
