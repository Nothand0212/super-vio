#include "super_vio/frame.hpp"

namespace super_vio
{
Frame::Frame( const std::size_t& id, const double& timestamp, const cv::Mat& left_image, const cv::Mat& right_image )
    : m_id( id ), m_timestamp( timestamp ), m_image_left( left_image.clone() ), m_image_right( right_image.clone() )
{
}

Frame::~Frame() = default;


void Frame::setImageLeft( const cv::Mat& image_left )
{
  this->m_image_left = image_left.clone();
}

cv::Mat Frame::getImageLeft() const
{
  return m_image_left;
}

void Frame::setImageRight( const cv::Mat& image_right )
{
  this->m_image_right = image_right.clone();
}

cv::Mat Frame::getImageRight() const
{
  return m_image_right;
}
std::size_t Frame::getId() const
{
  return m_id;
}
double Frame::getTimestamp() const
{
  return m_timestamp;
}

void Frame::addMapPoint( std::shared_ptr<MapPoint> map_point )
{
  m_map_points.push_back( map_point );
}

std::size_t Frame::getMapPointsCount() const
{
  return m_map_points.size();
}

void Frame::setData( const Features& features_left, const Features& features_right )
{
  m_key_points_left   = features_left.getKeyPoints();
  m_scores_left       = features_left.getScores();
  m_descriptors_left  = features_left.getDescriptors();
  m_key_points_right  = features_right.getKeyPoints();
  m_scores_right      = features_right.getScores();
  m_descriptors_right = features_right.getDescriptors();
}

// ****** Relate with Features ********
void Frame::setKeyPointsLeft( const std::vector<cv::Point2f>& key_points )
{
  m_key_points_left = key_points;
}
std::vector<cv::Point2f> Frame::getKeyPointsLeft() const
{
  return m_key_points_left;
}

void Frame::setKeyPointsRight( const std::vector<cv::Point2f>& key_points )
{
  m_key_points_right = key_points;
}
std::vector<cv::Point2f> Frame::getKeyPointsRight() const
{
  return m_key_points_right;
}
// **************************************

// ****** Relate with Scores ************
void Frame::setScoresLeft( const std::vector<float>& scores )
{
  m_scores_left = scores;
}
std::vector<float> Frame::getScoresLeft() const
{
  return m_scores_left;
}
void Frame::setScoresRight( const std::vector<float>& scores )
{
  m_scores_right = scores;
}
std::vector<float> Frame::getScoresRight() const
{
  return m_scores_right;
}
// **************************************


// ****** Relate with Descriptors ******
void Frame::setDescriptorsLeft( const cv::Mat& descriptors )
{
  m_descriptors_left = descriptors.clone();
}
cv::Mat Frame::getDescriptorsLeft() const
{
  return m_descriptors_left;
}
void Frame::setDescriptorsRight( const cv::Mat& descriptors )
{
  m_descriptors_right = descriptors.clone();
}
cv::Mat Frame::getDescriptorsRight() const
{
  return m_descriptors_right;
}
// **************************************

// ****** Relate with Outliers **********
void Frame::setIsOutlierLeft( const std::vector<bool>& is_outlier )
{
  m_is_outlier_left = is_outlier;
}
std::vector<bool> Frame::getIsOutlierLeft() const
{
  return m_is_outlier_left;
}
void Frame::setIsOutlierRight( const std::vector<bool>& is_outlier )
{
  m_is_outlier_right = is_outlier;
}
std::vector<bool> Frame::getIsOutlierRight() const
{
  return m_is_outlier_right;
}
// ******************************************

// ****** Relate with Pose ******************
void Frame::setPose( const Sophus::SE3d& pose )
{
  std::lock_guard<std::mutex> lock( m_mutex_pose );
  m_pose = pose;
}
Sophus::SE3d Frame::getPose()
{
  std::lock_guard<std::mutex> lock( m_mutex_pose );
  return m_pose;
}

void Frame::setPoseToRefferenceFrame( const Sophus::SE3d& pose_to_refference_frame )
{
  std::lock_guard<std::mutex> lock( m_mutex_pose_to_refference_frame );
  m_pose_to_refference_frame = pose_to_refference_frame;
}
Sophus::SE3d Frame::getPoseToRefferenceFrame()
{
  std::lock_guard<std::mutex> lock( m_mutex_pose_to_refference_frame );
  return m_pose_to_refference_frame;
}
// *****************************************
}  // namespace super_vio