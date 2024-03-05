/**
 ******************************************************************************
 * @file           : base_onnx_runner.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once
#include <onnxruntime_cxx_api.h>

#include "configuration.h"
#include "logger/logger.h"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/core/mat.hpp"
#include "opencv4/opencv2/core/types.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "vector"


class BaseOnnxRunner
{
public:
  // data
  // Ort::AllocatorWithDefaultOptions allocator_;
  enum IO : u_int8_t
  {
    INPUT  = 0,
    OUTPUT = 1
  };

  virtual int
  initOrtEnv( const Config& config )
  {
    return EXIT_SUCCESS;
  }

  virtual float getMatchThreshold()
  {
    return 0.0f;
  }

  virtual void setMatchThreshold( const float& threshold ) {}

  virtual double getTimer( const std::string& name )
  {
    return 0.0f;
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  inferenceImage( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst )
  {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  getKeyPointsResult()
  {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }

  void extractNodesInfo( const IO& io, std::vector<char*>& node_names, std::vector<std::vector<int64_t>>& node_shapes, const std::unique_ptr<Ort::Session>& session_uptr, const Ort::AllocatorWithDefaultOptions& allocator )
  {
    if ( io != IO::INPUT && io != IO::OUTPUT )
    {
      throw std::runtime_error( "io should be INPUT or OUTPUT" );
    }

    size_t num_nodes = ( io == IO::INPUT ) ? session_uptr->GetInputCount() : session_uptr->GetOutputCount();

    INFO( logger, "num_nodes: {0}", num_nodes );
    node_names.reserve( num_nodes );

    auto processNode = [ & ]( size_t i ) {
      char* node_name_temp = new char[ std::strlen( ( io == IO::INPUT ? session_uptr->GetInputNameAllocated( i, allocator ) : session_uptr->GetOutputNameAllocated( i, allocator ) ).get() ) + 1 ];
      std::strcpy( node_name_temp, ( io == IO::INPUT ? session_uptr->GetInputNameAllocated( i, allocator ) : session_uptr->GetOutputNameAllocated( i, allocator ) ).get() );
      INFO( logger, "extractor node name: {0}", node_name_temp );
      node_names.push_back( node_name_temp );
      node_shapes.emplace_back( ( io == IO::INPUT ? session_uptr->GetInputTypeInfo( i ) : session_uptr->GetOutputTypeInfo( i ) ).GetTensorTypeAndShapeInfo().GetShape() );
    };

    for ( size_t i = 0; i < num_nodes; i++ )
    {
      processNode( i );
    }
  }
};
