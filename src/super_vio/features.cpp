#include "super_vio/feature.hpp"

namespace super_vio
{
// ===============================================
// ==== ==== ==== ==== Feature ==== ==== ==== ====
// ===============================================

void Feature::operator=( const Feature& other )
{
  m_score        = other.m_score;
  m_key_point    = other.m_key_point;
  m_descriptor   = other.m_descriptor;
  m_wp_map_point = other.m_wp_map_point;
}

void Feature::setScore( const float& score )
{
  m_score = score;
}

void Feature::setKeyPoint( const cv::Point2f& key_point )
{
  m_key_point = key_point;
}

void Feature::setDescriptor( const cv::Mat& descriptor )
{
  m_descriptor = descriptor;
}

void Feature::setMapPoint( const std::shared_ptr<MapPoint>& map_point )
{
  // 判断传入的指针是否为 nullptr
  // if ( map_point )
  // {
  //   std::cout << "Setting MapPoint: " << map_point.get() << std::endl;
  // }
  // else
  // {
  //   std::cout << "Received a nullptr for MapPoint" << std::endl;
  // }

  m_wp_map_point = map_point;

  // 打印 m_wp_map_point 的引用计数
  // std::cout << "Current reference count after setting: " << m_wp_map_point.use_count() << std::endl;
}

float Feature::getScore() const
{
  return m_score;
}

cv::Point2f Feature::getKeyPoint() const
{
  return m_key_point;
}

cv::Mat Feature::getDescriptor() const
{
  return m_descriptor;
}

std::shared_ptr<MapPoint> Feature::getMapPoint() const
{
  std::cout << "Feature::getMapPoint() use_count: " << m_wp_map_point.use_count() << std::endl;
  return m_wp_map_point;
}

// ================================================
// ==== ==== ==== ==== Features ==== ==== ==== ====
// ================================================

Features::Features( std::vector<float> scores, std::vector<cv::Point2f> key_points, cv::Mat descriptors )
{
  if ( scores.size() == key_points.size() && key_points.size() == descriptors.rows )
  {
    for ( int i = 0; i < scores.size(); i++ )
    {
      Feature feature;
      feature.setScore( scores[ i ] );
      feature.setKeyPoint( key_points[ i ] );
      feature.setDescriptor( descriptors.row( i ) );
      m_v_features.push_back( feature );
    }
  }
  else
  {
    std::cerr << "Error: scores, key_points and descriptors size not match. Construct a Null Features." << std::endl;
    m_v_features = std::vector<Feature>();
  }
}

std::vector<Feature> Features::getFeatures() const
{
  return m_v_features;
}

std::vector<float> Features::getScores() const
{
  std::vector<float> scores;
  for ( const auto& feature : m_v_features )
  {
    scores.push_back( feature.getScore() );
  }
  return scores;
}

std::vector<cv::Point2f> Features::getKeyPoints() const
{
  std::vector<cv::Point2f> key_points;
  for ( const auto& feature : m_v_features )
  {
    key_points.push_back( feature.getKeyPoint() );
  }
  return key_points;
}

cv::Mat Features::getDescriptors() const
{
  cv::Mat descriptors;
  descriptors.reserve( m_v_features.size() );
  for ( const auto& feature : m_v_features )
  {
    descriptors.push_back( feature.getDescriptor() );
  }
  return descriptors;
}

std::vector<std::shared_ptr<MapPoint>> Features::getMapPoints() const
{
  std::vector<std::shared_ptr<MapPoint>> map_points;
  for ( const auto& feature : m_v_features )
  {
    map_points.push_back( feature.getMapPoint() );
  }
  return map_points;
}


float Features::getSingleScore( const std::size_t& idx ) const
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].getScore();
}

cv::Point2f Features::getSingleKeyPoint( const std::size_t& idx ) const
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].getKeyPoint();
}

cv::Mat Features::getSingleDescriptor( const std::size_t& idx ) const
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].getDescriptor();
}

std::shared_ptr<MapPoint> Features::getSingleMapPoint( const std::size_t& idx ) const
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].getMapPoint();
}


void Features::setSingleScore( const std::size_t& idx, const float& score )
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].setScore( score );
}

void Features::setSingleKeyPoint( const std::size_t& idx, const cv::Point2f& key_point )
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].setKeyPoint( key_point );
}

void Features::setSingleDescriptor( const std::size_t& idx, const cv::Mat& descriptor )
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].setDescriptor( descriptor );
}

void Features::setSingleMapPoint( const std::size_t& idx, const std::shared_ptr<MapPoint>& map_point )
{
  assert( idx < m_v_features.size() );
  return m_v_features[ idx ].setMapPoint( map_point );
}


}  // namespace super_vio