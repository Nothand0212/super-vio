/**
 ******************************************************************************
 * @file           : include/extractor/extractor.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : The key points extractor with onnx model
 * @attention      : None
 * @date           : 24-1-25
 ******************************************************************************
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "super_vio/base_onnx_runner.h"
#include "super_vio/feature.hpp"
#include "utilities/configuration.h"
// #include "data/features.h"

#include "super_vio/frame.h"
#include "utilities/accumulate_average.h"
#include "utilities/color.h"
#include "utilities/image_process.h"
#include "utilities/timer.h"
namespace super_vio
{
inline std::vector<cv::Point2f> getKeyPointsInOriginalImage( const std::vector<cv::Point2f>& key_points, const float scale )
{
  std::vector<cv::Point2f> key_points_in_original_image;
  for ( const auto& key_point : key_points )
  {
    key_points_in_original_image.emplace_back( cv::Point2f( ( key_point.x ) / scale, ( key_point.y ) / scale ) );
    // key_points_in_original_image.emplace_back( cv::Point2f( ( key_point.x + 0.5f ) / scale, ( key_point.y + 0.5f ) / scale ) );
  }
  return key_points_in_original_image;
}

class Extractor : public BaseOnnxRunner
{
private:
  unsigned int m_threads_num;

  Ort::Env                      m_env;
  Ort::SessionOptions           m_session_options;
  std::unique_ptr<Ort::Session> m_uptr_session;

  Ort::AllocatorWithDefaultOptions m_allocator;

  std::vector<char*>                m_vec_input_names;
  std::vector<std::vector<int64_t>> m_vec_input_shapes;

  std::vector<char*>                m_vec_output_names;
  std::vector<std::vector<int64_t>> m_vec_output_shapes;

  Features m_key_points;

  utilities::Timer             m_timer;
  utilities::AccumulateAverage m_average_timer;

  //   std::vector<float> m_scale{ 1.0f, 1.0f };
  float        m_scale{ 1.0f };
  int          m_height_transformed{ 0 };
  int          m_width_transformed{ 0 };
  unsigned int m_point_num{ 0 };

private:
  cv::Mat  prePorcess( const utilities::Configuration& config, const cv::Mat& image, float& scale );
  int      inference( const utilities::Configuration& config, const cv::Mat& image );
  Features postProcess( const utilities::Configuration& config, std::vector<Ort::Value> tensor );


public:
  explicit Extractor( unsigned int threads_num = 0, unsigned int point_num = 0 );
  ~Extractor();

  int initOrtEnv( const utilities::Configuration& config ) override;

  Features inferenceImage( const utilities::Configuration& config, const cv::Mat& image );
  Features getKeyPoints() const;
  float    getScale() const;
  int      getHeightTransformed() const;
  int      getWidthTransformed() const;

  Features                                 distributeKeyPoints( const Features& key_points, const cv::Mat& image );
  std::pair<Features, std::vector<Region>> distributeKeyPointsDebug( const Features& key_points, const cv::Mat& image );
};
}  // namespace super_vio