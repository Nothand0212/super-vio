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

class Features
{
private:
  std::vector<cv::Point2f> m_v_key_points;
  std::vector<float>       m_v_scores;
  cv::Mat                  m_mat_descriptor;
  std::weak_ptr<MapPoint>  m_wp_map_point;

public:
  Features()  = default;
  ~Features() = default;

  void operator=( const Features& features );

  void setKeyPoints( const std::vector<cv::Point2f>& key_points );

  void                      setMapPoint( const std::shared_ptr<MapPoint>& map_point );
  std::shared_ptr<MapPoint> getMapPoint();

  void setScores( const std::vector<float>& scores );

  void setDescriptor( const cv::Mat& descriptor );

  std::vector<cv::Point2f> getKeyPoints() const;

  std::vector<float> getScores() const;

  cv::Mat getDescriptor() const;
};
}  // namespace super_vio