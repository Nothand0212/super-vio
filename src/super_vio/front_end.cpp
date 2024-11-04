#include "super_vio/front_end.hpp"

namespace super_vio
{
FrontEnd::FrontEnd( const utilities::Configuration& config ) : config_{ config }
{
}

FrontEnd::~FrontEnd()
{
}

bool FrontEnd::setStereoCamera( std::shared_ptr<Camera> left_camera, std::shared_ptr<Camera> right_camera )
{
  if ( left_camera == nullptr || right_camera == nullptr )
  {
    throw std::invalid_argument( "Camera pointers cannot be null" );
    ERROR( super_vio::logger, "Camera pointers cannot be null" );
    exit( -1 );
  }

  this->left_camera_  = left_camera;
  this->right_camera_ = right_camera;

  return true;
}

bool FrontEnd::setExtractor( std::shared_ptr<Extractor> extractor )
{
  if ( extractor == nullptr )
  {
    throw std::invalid_argument( "Extractor pointer cannot be null" );
    ERROR( super_vio::logger, "Extractor pointer cannot be null" );
    return false;
  }

  this->extractor_ = extractor;

  return true;
}

bool FrontEnd::setMatcher( std::shared_ptr<Matcher> matcher )
{
  if ( matcher == nullptr )
  {
    throw std::invalid_argument( "Matcher pointer cannot be null" );
    ERROR( super_vio::logger, "Matcher pointer cannot be null" );
    return false;
  }

  this->matcher_ = matcher;
  return true;
}

bool FrontEnd::updateStereoImages( const cv::Mat& left_img, const cv::Mat& right_img, const double& timestamp )
{
  this->current_frame_ = std::make_shared<Frame>( this->frame_id_, timestamp, left_img, right_img );

  // TODO: add Undistort Function later but kitti no need to do

  // Tracking Stage
  {
    switch ( this->track_status_ )
    {
      case FrontEndStatus::INITIALIZING:
      {
        return this->buildInitialMap();
        break;
      }

      case FrontEndStatus::TRACKING_GOOD:
      {
        return trackBundleAdjust();
        break;
      }

      case FrontEndStatus::TRACKING_BAD:
      {
        return trackIterativeClosestPoint();
        break;
      }

      case FrontEndStatus::LOST:
      {
        // TODO: Add Relocalization Function or re-build init map to start a whole new map
        return false;
      }
    }
  }
  return false;
}

bool FrontEnd::buildInitialMap()
{
  utilities::Timer test_timer;

  test_timer.tic();
  auto features_on_left_img  = this->extractor_->inferenceImage( this->config_, this->current_frame_->getImageLeft() );
  auto features_on_right_img = this->extractor_->inferenceImage( this->config_, this->current_frame_->getImageRight() );

  INFO( super_vio::logger, "Both Inference time: {0}", test_timer.tocGetDuration() );


  auto key_points_left   = features_on_left_img.getKeyPoints();
  auto key_points_right  = features_on_right_img.getKeyPoints();
  auto descriptors_left  = features_on_left_img.getDescriptors();
  auto descriptors_right = features_on_right_img.getDescriptors();

  float scale_temp = this->extractor_->getScale();

  this->matcher_->setParams( std::vector<float>( scale_temp, scale_temp ), this->extractor_->getHeightTransformed(), this->extractor_->getWidthTransformed(), 0.0f );
  std::set<std::pair<int, int>> matches_set = this->matcher_->inferenceDescriptorPair( this->config_, key_points_left, key_points_right, descriptors_left, descriptors_right );

  std::vector<cv::Point2f> key_points_transformed_src = getKeyPointsInOriginalImage( key_points_left, scale_temp );
  std::vector<cv::Point2f> key_points_transformed_dst = getKeyPointsInOriginalImage( key_points_right, scale_temp );

  std::vector<cv::Point2f> matches_src;
  std::vector<cv::Point2f> matches_dst;
  for ( const auto& match : matches_set )
  {
    matches_src.emplace_back( key_points_transformed_src[ match.first ] );
    matches_dst.emplace_back( key_points_transformed_dst[ match.second ] );
  }
  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_pair = std::make_pair( matches_src, matches_dst );

  test_timer.tic();
  std::size_t success_count = 0;
  for ( const auto& match : matches_set )
  {
    Eigen::Vector3d point_3d = Eigen::Vector3d::Zero();

    bool success = utilities::compute3DPoint( this->left_camera_->getParams(), this->right_camera_->getParams(),
                                              this->left_camera_->getPoseMatrix(), this->right_camera_->getPoseMatrix(),
                                              key_points_transformed_src[ match.first ], key_points_transformed_dst[ match.second ], point_3d );

    if ( !success )
    {
      // WARN( super_vio::logger, "Triangulate failed" );
      continue;
    }
    else
    {
      // Test Map Point
      std::shared_ptr<super_vio::MapPoint> map_point_ptr( new super_vio::MapPoint );
      map_point_ptr->setPosition( point_3d );

      features_on_left_img.setSingleMapPoint( match.first, map_point_ptr );
      features_on_right_img.setSingleMapPoint( match.second, map_point_ptr );

      success_count++;
    }
  }
  INFO( super_vio::logger, "Triangulate {} / {} --> Time Consumed: {}", success_count, matches_set.size(), test_timer.tocGetDuration() );

  // TODO: Add Map Class

  return true;
}

bool FrontEnd::trackBundleAdjust()
{
  return false;
}

bool FrontEnd::trackIterativeClosestPoint()
{
  return false;
}

}  // namespace super_vio