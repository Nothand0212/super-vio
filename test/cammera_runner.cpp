#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "base_onnx_runner.h"
#include "camera.h"
#include "configuration.h"
#include "extractor/extractor.h"
#include "frame.h"
#include "image_process.h"
#include "logger/logger.h"
#include "matcher/matcher.h"
#include "utilities/accumulate_average.h"
#include "utilities/reconstructor.h"
#include "utilities/timer.h"
#include "visualizer.h"

int main()
{
  // initialize logger
  InitLogger( "/home/lin/Projects/super-vio/log_data/tmp.log" );
  INFO( logger, "Start" );

  // initialize config
  Config cfg{};
  cfg.readConfig( "/home/lin/Projects/super-vio/config/param.json" );

  //intialize extractor
  Extracotr *extractor_left_ptr, *extractor_right_ptr;
  extractor_left_ptr = new Extracotr{ 6, 200 };
  extractor_left_ptr->initOrtEnv( cfg );
  extractor_right_ptr = new Extracotr{ 6, 200 };
  extractor_right_ptr->initOrtEnv( cfg );

  // matcher init
  std::unique_ptr<Matcher> matcher_ptr = std::make_unique<Matcher>();
  matcher_ptr->initOrtEnv( cfg );

  CameraDriver camera;

  cv::Mat           image_left, image_right;
  Timer             timer;
  AccumulateAverage accumulate_average_timer;
  double            time_stamp    = 0.0;
  double            time_consumed = 0.0;
  std::size_t       frame_id      = 0;

  while ( true )
  {
    timer.tic();

    camera.getStereoFrame( image_left, image_right );

    time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() ).count() / 1000.0;

    if ( image_left.empty() || image_right.empty() )
    {
      WARN( logger, "Empty image. Left: {0}, Right: {1}", image_left.empty(), image_right.empty() );
      continue;
    }

    auto left_future = std::async( std::launch::async, [ extractor_left_ptr, cfg, image_left ]() {
      return extractor_left_ptr->inferenceImage( cfg, image_left );
    } );

    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, cfg, image_right ]() {
      return extractor_right_ptr->inferenceImage( cfg, image_right );
    } );

    auto extract_result_left  = left_future.get();
    auto extract_result_right = right_future.get();

    Frame current_frame( frame_id, time_stamp, image_left, image_right );
    current_frame.setData( extract_result_left, extract_result_right );

    // Matching Stereo Pair
    float scale_temp = extractor_left_ptr->getScale();
    matcher_ptr->setParams( std::vector<float>( scale_temp, scale_temp ), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f );
    auto matches_set = matcher_ptr->inferenceDescriptorPair( cfg, current_frame.getKeyPointsLeft(), current_frame.getKeyPointsRight(), current_frame.getDescriptorsLeft(), current_frame.getDescriptorsRight() );


    // Triangulate keypoints
    Eigen::Matrix<double, 3, 4> pose_left, pose_right;
    for ( const auto &match : matches_set )
    {
      Eigen::Vector3d point_3d;

      bool success = triangulate( pose_left, pose_right, current_frame.getKeyPointsLeft()[ match.first ], current_frame.getKeyPointsRight()[ match.second ], point_3d );

      if ( !success )
      {
        WARN( logger, "Triangulate failed" );
        continue;
      }
      else if ( point_3d[ 2 ] < 0 )
      {
        WARN( logger, "Triangulate failed, point behind camera" );
        continue;
      }
      else
      {
        INFO( logger, "Triangulate success. Point: [{0}, {1}, {2}]", point_3d[ 0 ], point_3d[ 1 ], point_3d[ 2 ] );
        std::size_t index = current_frame.getMapPointsCount();

        if ( index < 0 )
        {
          WARN( logger, "Index is negative" );
        }
        else
        {
          INFO( logger, "Adding MapPoint." );
          std::shared_ptr<MapPoint> mp;
          mp->setPosition( point_3d );
          mp->addObservation( &current_frame, index );
          current_frame.addMapPoint( mp );
        }
      }
    }

    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( logger, "Frame {0} time consumed: {1} ms, average time consumed: {2} ms", frame_id, time_consumed, accumulate_average_timer.getAverage() );
    visualizeKeyPoints( current_frame.getImageLeft(), current_frame.getImageRight(), current_frame.getKeyPointsLeft(), current_frame.getKeyPointsRight() );


    frame_id++;
  }
}