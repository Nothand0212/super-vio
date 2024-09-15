#pragma once
#define SOPHUS_USE_BASIC_LOGGING

#include <eigen3/Eigen/Core>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "super_vio/feature.hpp"
#include "super_vio/map_point.hpp"

namespace super_vio
{
class MapPoint;  // forward declaration
class Features;
class Frame
{
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // this member variable is update by Extractor
  std::vector<cv::Point2f> m_key_points_left;
  std::vector<float>       m_scores_left;
  cv::Mat                  m_descriptors_left;
  std::vector<cv::Point2f> m_key_points_right;
  std::vector<float>       m_scores_right;
  cv::Mat                  m_descriptors_right;

  // MapPoint
  std::vector<std::shared_ptr<MapPoint>> m_map_points;

  // this member variable is update on PnP solving stage
  std::vector<bool> m_is_outlier_left;
  std::vector<bool> m_is_outlier_right;
  Sophus::SE3d      m_pose;                      // pose of the camera in the world frame
  Sophus::SE3d      m_pose_to_refference_frame;  // pose of the camera in the refference frame
  std::mutex        m_mutex_pose;
  std::mutex        m_mutex_pose_to_refference_frame;

  // this member variable is update on constructing the frame
  std::size_t m_id;
  double      m_timestamp;
  cv::Mat     m_image_left, m_image_right;

public:
  Frame( const std::size_t& id, const double& timestamp, const cv::Mat& left_image, const cv::Mat& right_image );
  ~Frame();

  cv::Mat     getImageLeft() const;
  cv::Mat     getImageRight() const;
  std::size_t getId() const;
  double      getTimestamp() const;

  void addMapPoint( std::shared_ptr<MapPoint> map_point );

  std::size_t getMapPointsCount() const;

  void setData( const Features& features_left, const Features& features_right );

  // ****** Relate with Features ********
  void                     setKeyPointsLeft( const std::vector<cv::Point2f>& key_points );
  std::vector<cv::Point2f> getKeyPointsLeft() const;
  void                     setKeyPointsRight( const std::vector<cv::Point2f>& key_points );
  std::vector<cv::Point2f> getKeyPointsRight() const;
  // **************************************

  // ****** Relate with Scores ************
  void               setScoresLeft( const std::vector<float>& scores );
  std::vector<float> getScoresLeft() const;
  void               setScoresRight( const std::vector<float>& scores );
  std::vector<float> getScoresRight() const;
  // **************************************


  // ****** Relate with Descriptors ******
  void    setDescriptorsLeft( const cv::Mat& descriptors );
  cv::Mat getDescriptorsLeft() const;
  void    setDescriptorsRight( const cv::Mat& descriptors );
  cv::Mat getDescriptorsRight() const;
  // **************************************

  // ****** Relate with Outliers **********
  void              setIsOutlierLeft( const std::vector<bool>& is_outlier );
  std::vector<bool> getIsOutlierLeft() const;
  void              setIsOutlierRight( const std::vector<bool>& is_outlier );
  std::vector<bool> getIsOutlierRight() const;
  // ******************************************

  // ****** Relate with Pose ******************
  void         setPose( const Sophus::SE3d& pose );
  Sophus::SE3d getPose();

  void         setPoseToRefferenceFrame( const Sophus::SE3d& pose_to_refference_frame );
  Sophus::SE3d getPoseToRefferenceFrame();
  // *****************************************
};
}  // namespace super_vio