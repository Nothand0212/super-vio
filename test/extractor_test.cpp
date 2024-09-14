#include <future>
#include <opencv2/opencv.hpp>
#include <thread>

#include "logger/logger.h"
#include "super_vio/base_onnx_runner.h"
#include "super_vio/extractor.h"
#include "utilities/accumulate_average.h"
#include "utilities/configuration.h"
#include "utilities/image_process.h"
#include "utilities/timer.h"
#include "utilities/visualizer.h"

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
  super_vio::initLogger( "/home/lin/CLionProjects/light_glue_onnx/log/tmp.log" );
  INFO( super_vio::logger, "Start" );

  utilities::Timer             timer;
  utilities::AccumulateAverage accumulate_average_timer;

  utilities::Configuration cfg{};
  cfg.readConfigFile( "/home/lin/CLionProjects/light_glue_onnx/config/param.json" );

  std::vector<cv::String> image_file_src_vec;
  std::vector<cv::String> image_file_dst_vec;

  // Read image file path
  cv::glob( cfg.image_src_path, image_file_src_vec );
  cv::glob( cfg.image_dst_path, image_file_dst_vec );

  // Read image
  if ( image_file_src_vec.size() != image_file_dst_vec.size() )
  {
    ERROR( super_vio::logger, "image src number: {0}", image_file_src_vec.size() );
    ERROR( super_vio::logger, "image dst number: {0}", image_file_dst_vec.size() );
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_src_mat_vec = readImage( image_file_src_vec, cfg.gray_flag );
  std::vector<cv::Mat> image_dst_mat_vec = readImage( image_file_dst_vec, cfg.gray_flag );

  // end2end
  Extractor *extractor_left_ptr, *extractor_right_ptr;
  extractor_left_ptr = new Extractor{ 6, 200 };
  extractor_left_ptr->initOrtEnv( cfg );
  extractor_right_ptr = new Extractor{ 6, 200 };
  extractor_right_ptr->initOrtEnv( cfg );


  // inference
  int    count = 0;
  double time_consumed;
  auto   iter_src = image_src_mat_vec.begin();
  auto   iter_dst = image_dst_mat_vec.begin();
  for ( ; iter_src != image_src_mat_vec.end(); ++iter_src, ++iter_dst )
  {
    INFO( super_vio::logger, "processing image {0} / {1}", image_file_src_vec[ count ], image_file_dst_vec[ count ] );
    count++;
    timer.tic();

    auto left_future = std::async( std::launch::async, [ extractor_left_ptr, cfg, iter_src ]() {
      return extractor_left_ptr->inferenceImage( cfg, *iter_src );
    } );

    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, cfg, iter_dst ]() {
      return extractor_right_ptr->inferenceImage( cfg, *iter_dst );
    } );

    auto key_points_result_left  = left_future.get();
    auto key_points_result_right = right_future.get();

    // Features key_points_result_left  = extractor_left_ptr->inferenceImage( cfg, *iter_src );
    // Features key_points_result_right = extractor_right_ptr->inferenceImage( cfg, *iter_dst );


    auto key_points_src = key_points_result_left.getKeyPoints();
    auto key_points_dst = key_points_result_right.getKeyPoints();


    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( super_vio::logger, "time consumed: {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );
    INFO( super_vio::logger, "key points number: {0} / {1}", key_points_src.size(), key_points_dst.size() );

    visualizeKeyPoints( *iter_src, *iter_dst, key_points_src, key_points_dst );

    timer.tic();
    auto key_points_src_distribute = extractor_left_ptr->distributeKeyPointsDebug( key_points_result_left, *iter_src );
    auto key_points_dst_distribute = extractor_right_ptr->distributeKeyPointsDebug( key_points_result_right, *iter_dst );
    INFO( super_vio::logger, "key points number: {0} / {1}", key_points_src_distribute.first.getKeyPoints().size(), key_points_dst_distribute.first.getKeyPoints().size() );
    time_consumed = timer.tocGetDuration();
    INFO( super_vio::logger, "distribute time consumed: {0} ", time_consumed );

    visualizeKeyPoints( *iter_src, *iter_dst, key_points_src_distribute.first.getKeyPoints(), key_points_dst_distribute.first.getKeyPoints(), key_points_src_distribute.second, key_points_dst_distribute.second );
  }
  return 0;
}
