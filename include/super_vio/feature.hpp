/**
 * @file feature.hpp
 * @author LinZeshi (linzeshi@foxmail.com)
 * @brief A feature poins class for Super-VIO.
 * @version 0.1
 * @date 2024-09-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "super_vio/map_point.hpp"
namespace super_vio
{
class MapPoint;
class Region
{
public:
  cv::Rect2i      rectangle;
  std::deque<int> indexs;
};

class Feature
{
private:
  float m_score;

  cv::Point2f               m_key_point;
  cv::Mat                   m_descriptor;
  std::shared_ptr<MapPoint> m_wp_map_point;

public:
  Feature()    = default;
  ~Feature()   = default;
  void operator=( const Feature& feature );

  void setScore( const float& score );
  void setKeyPoint( const cv::Point2f& key_point );
  void setDescriptor( const cv::Mat& descriptor );
  void setMapPoint( const std::shared_ptr<MapPoint>& map_point );

  float                     getScore() const;
  cv::Point2f               getKeyPoint() const;
  cv::Mat                   getDescriptor() const;
  std::shared_ptr<MapPoint> getMapPoint() const;
};


class Features
{
private:
  std::vector<Feature> m_v_features;

public:
  Features() = default;
  Features( std::vector<float> scores, std::vector<cv::Point2f> key_points, cv::Mat descriptors );

  ~Features() = default;


  std::vector<Feature> getFeatures() const;

  std::vector<float>                     getScores() const;
  std::vector<cv::Point2f>               getKeyPoints() const;
  cv::Mat                                getDescriptors() const;
  std::vector<std::shared_ptr<MapPoint>> getMapPoints() const;

  float                     getSingleScore( const std::size_t& idx ) const;
  cv::Point2f               getSingleKeyPoint( const std::size_t& idx ) const;
  cv::Mat                   getSingleDescriptor( const std::size_t& idx ) const;
  std::shared_ptr<MapPoint> getSingleMapPoint( const std::size_t& idx ) const;

  void setSingleScore( const std::size_t& idx, const float& score );
  void setSingleKeyPoint( const std::size_t& idx, const cv::Point2f& key_point );
  void setSingleDescriptor( const std::size_t& idx, const cv::Mat& descriptor );
  void setSingleMapPoint( const std::size_t& idx, const std::shared_ptr<MapPoint>& map_point );
};
}  // namespace super_vio