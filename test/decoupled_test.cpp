#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "configuration.h"
#include "decoupled_onnx_runner/decoupled_onxx_runner.h"
#include "image_process.h"
#include "logger/logger.h"
#include "utilities/accumulate_average.h"
#include "utilities/timer.h"
#include "visualizer.h"


std::vector<cv::Mat> readImage( std::vector<cv::String> image_file_vec, bool grayscale = false )
{
  /*
    Func:
        Read an image from path as RGB or grayscale

    */
  int mode = cv::IMREAD_COLOR;
  if ( grayscale )
  {
    mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
  }

  std::vector<cv::Mat> image_matlist;
  for ( const auto& file : image_file_vec )
  {
    cv::Mat image = cv::imread( file, mode );
    if ( image.empty() )
    {
      throw std::runtime_error( "[ERROR] Could not read image at " + file );
    }
    if ( !grayscale )
    {
      cv::cvtColor( image, image, cv::COLOR_BGR2RGB );  // BGR -> RGB
    }
    image_matlist.emplace_back( image );
  }

  return image_matlist;
}


int main( int argc, char const* argv[] )
{
  InitLogger( "/home/lin/CLionProjects/light_glue_onnx/log/tmp.log" );
  INFO( logger, "Start" );

  Timer             timer;
  AccumulateAverage accumulate_average_timer;

  Config cfg{};
  cfg.readConfig( "/home/lin/CLionProjects/light_glue_onnx/config/param.json" );

  std::vector<cv::String> image_file_src_vec;
  std::vector<cv::String> image_file_dst_vec;

  // Read image file path
  cv::glob( cfg.image_src_path, image_file_src_vec );
  cv::glob( cfg.image_dst_path, image_file_dst_vec );

  // Read image
  if ( image_file_src_vec.size() != image_file_dst_vec.size() )
  {
    ERROR( logger, "image src number: {0}", image_file_src_vec.size() );
    ERROR( logger, "image dst number: {0}", image_file_dst_vec.size() );
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_src_mat_vec = readImage( image_file_src_vec, cfg.gray_flag );
  std::vector<cv::Mat> image_dst_mat_vec = readImage( image_file_dst_vec, cfg.gray_flag );

  // end2end
  DecoupledOnnxRunner* feature_matcher;
  feature_matcher = new DecoupledOnnxRunner{ 0 };
  feature_matcher->initOrtEnv( cfg );
  feature_matcher->setMatchThreshold( cfg.threshold );

  // inference
  int    count = 0;
  double time_consumed;
  auto   iter_src = image_src_mat_vec.begin();
  auto   iter_dst = image_dst_mat_vec.begin();
  for ( ; iter_src != image_src_mat_vec.end(); ++iter_src, ++iter_dst )
  {
    INFO( logger, "processing image {0} / {1}", image_file_src_vec[ count ], image_file_dst_vec[ count ] );
    count++;
    timer.tic();
    auto key_points_result = feature_matcher->inferenceImagePair( cfg, *iter_src, *iter_dst );
    time_consumed          = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( logger, "time consumed: {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );

    auto key_points_src = feature_matcher->getKeyPointsSrc();
    auto key_points_dst = feature_matcher->getKeyPointsDst();

    visualizeMatches( *iter_src, *iter_dst, key_points_result, key_points_src, key_points_dst );
  }
  return 0;
}
