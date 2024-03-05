#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "frame.h"
class Frame;  // forward declaration

class MapPoint
{
private:
  std::size_t m_id;

  std::mutex      m_mutex_position;
  Eigen::Vector3d m_position;


  std::mutex                    m_mutex_observations;
  std::size_t                   m_observations_count;
  std::map<Frame*, std::size_t> m_observations;

public:
  MapPoint( /* args */ );
  ~MapPoint();

  void            setPosition( const Eigen::Vector3d& position );
  Eigen::Vector3d getPosition();

  void addObservation( Frame* frame, std::size_t idx );
};
