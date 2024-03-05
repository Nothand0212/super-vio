/**
 ******************************************************************************
 * @file           : include/matcher/matcher.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : The key points matcher with onnx model
 * @attention      : None
 * @date           : 24-1-28
 ******************************************************************************
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "utilities/accumulate_average.h"
#include "utilities/color.h"
#include "utilities/timer.h"

class Matcher : public BaseOnnxRunner
{
private:
  unsigned int       m_threads_num;
  float              m_match_threshold_{ 0.0f };
  std::vector<float> m_vec_sacles{ 1.0f, 1.0f };
  int                m_height{ -1 };
  int                m_width{ -1 };

  //   std::vector<cv::Point2f>      m_key_points_src;
  //   std::vector<cv::Point2f>      m_key_points_dst;
  std::set<std::pair<int, int>> m_matched_indices;

  Ort::Env                         m_env;
  Ort::SessionOptions              m_session_options;
  std::unique_ptr<Ort::Session>    m_uptr_session;
  Ort::AllocatorWithDefaultOptions m_allocator;

  std::vector<char*>                m_vec_input_names;
  std::vector<std::vector<int64_t>> m_vec_input_shapes;

  std::vector<char*>                m_vec_output_names;
  std::vector<std::vector<int64_t>> m_vec_output_shapes;

  std::vector<Ort::Value> m_vec_output_tensor;

  Timer             m_timer;
  AccumulateAverage m_average_timer;

private:
  std::vector<cv::Point2f> preProcess( std::vector<cv::Point2f> key_points, const int& height, const int& width );
  int                      inference( const Config& config, const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst );
  int                      postProcess( const Config& config );

public:
  Matcher( unsigned int threads_num = 0 );
  ~Matcher();


  int initOrtEnv( const Config& config );

  void  setParams( const std::vector<float>& scales, const int& height, const int& width, const float& threshold );
  void  setScales( const std::vector<float>& vec_scales );
  void  setHeight( const int& height );
  void  setWidth( const int& width );
  void  setMatchThreshold( const float& threshold );
  float getMatchThreshold() const;

  std::set<std::pair<int, int>> inferenceDescriptorPair( const Config& config, const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst );
};
