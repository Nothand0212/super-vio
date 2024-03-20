#include "super_vio/frame.h"
#include "super_vio/mapPoint.h"

namespace super_vio
{
MapPoint::MapPoint( /* args */ )
{
  static std::size_t factory_id = 0;

  m_id = factory_id++;
}

MapPoint::~MapPoint()
{
}

void MapPoint::setPosition( const Eigen::Vector3d& position )
{
  std::lock_guard<std::mutex> lock( m_mutex_position );
  m_position = position;
}

Eigen::Vector3d MapPoint::getPosition()
{
  std::lock_guard<std::mutex> lock( m_mutex_position );
  return m_position;
}

// this MapPoint is observed by frame at index idx
// frame has a member variable "std::vector<MapPoint> m_map_points", idx is the index of this MapPoint in that vector
void MapPoint::addObservation( Frame* frame, std::size_t idx )
{
  if ( m_observations.count( frame ) )
  {
    // already observed by this frame, do nothing
    return;
  }
  else
  {
    m_observations[ frame ] = idx;
    m_observations_count++;
  }
}
}  // namespace super_vio