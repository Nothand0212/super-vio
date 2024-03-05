#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "base_onnx_runner.h"
#include "configuration.h"
#include "extractor/extractor.h"
#include "image_process.h"
#include "logger/logger.h"
#include "matcher/matcher.h"
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

  std::shared_ptr<Extracotr> extractor_left_ptr = std::make_unique<Extracotr>( 6, 200 );
  extractor_left_ptr->initOrtEnv( cfg );
  std::shared_ptr<Extracotr> extractor_right_ptr = std::make_unique<Extracotr>( 6, 200 );
  extractor_right_ptr->initOrtEnv( cfg );

  // matcher init
  std::unique_ptr<Matcher> matcher_ptr = std::make_unique<Matcher>();
  matcher_ptr->initOrtEnv( cfg );

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

    auto left_future = std::async( std::launch::async, [ extractor_left_ptr, cfg, iter_src ]() {
      return extractor_left_ptr->inferenceImage( cfg, *iter_src );
    } );

    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, cfg, iter_dst ]() {
      return extractor_right_ptr->inferenceImage( cfg, *iter_dst );
    } );

    auto key_points_result_left  = left_future.get();
    auto key_points_result_right = right_future.get();

    auto key_points_src = key_points_result_left.getKeyPoints();
    auto key_points_dst = key_points_result_right.getKeyPoints();

    float scale_temp = extractor_left_ptr->getScale();
    matcher_ptr->setParams( std::vector<float>( scale_temp, scale_temp ), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f );
    auto matches_set = matcher_ptr->inferenceDescriptorPair( cfg, key_points_src, key_points_dst, key_points_result_left.getDescriptor(), key_points_result_right.getDescriptor() );


    std::vector<cv::Point2f> matches_src;
    std::vector<cv::Point2f> matches_dst;
    for ( const auto& match : matches_set )
    {
      matches_src.emplace_back( key_points_src[ match.first ] );
      matches_dst.emplace_back( key_points_dst[ match.second ] );
    }
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_pair = std::make_pair( matches_src, matches_dst );

    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( logger, "time consumed: {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );
    INFO( logger, "key points number: {0} / {1}", key_points_src.size(), key_points_dst.size() );

    // visualizeKeyPoints( *iter_src, *iter_dst, key_points_src, key_points_dst );
    visualizeMatches( *iter_src, *iter_dst, matches_pair, key_points_src, key_points_dst );
  }
  return 0;
}
