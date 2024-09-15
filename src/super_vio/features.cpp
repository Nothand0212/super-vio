#include "super_vio/feature.hpp"

namespace super_vio
{
void Features::operator=( const Features& features )
{
  m_v_key_points   = features.m_v_key_points;
  m_v_scores       = features.m_v_scores;
  m_mat_descriptor = features.m_mat_descriptor.clone();
}

void Features::setKeyPoints( const std::vector<cv::Point2f>& key_points )
{
  m_v_key_points = key_points;
}

void Features::setMapPoint( const std::shared_ptr<MapPoint>& map_point )
{
  m_wp_map_point = map_point;
}

std::shared_ptr<MapPoint> Features::getMapPoint()
{
  return m_wp_map_point.lock();
}


void Features::setScores( const std::vector<float>& scores )
{
  m_v_scores = scores;
}

void Features::setDescriptor( const cv::Mat& descriptor )
{
  m_mat_descriptor = descriptor.clone();
}

std::vector<cv::Point2f> Features::getKeyPoints() const
{
  return m_v_key_points;
}

std::vector<float> Features::getScores() const
{
  return m_v_scores;
}

cv::Mat Features::getDescriptor() const
{
  return m_mat_descriptor;
}
}  // namespace super_vio