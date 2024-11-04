#pragma once

#include <cstdint>

#include "logger/mine_logger.hpp"
#include "super_vio/base_onnx_runner.hpp"
#include "super_vio/camera.hpp"
#include "super_vio/extractor.hpp"
#include "super_vio/frame.hpp"
#include "super_vio/map_point.hpp"
#include "super_vio/matcher.hpp"
#include "super_vio/pose_estimator_3d3d.hpp"
#include "super_vio/pose_graph_optimizer.hpp"
#include "super_vio/ros_tool.hpp"
#include "utilities/accumulate_average.hpp"
#include "utilities/color.hpp"
#include "utilities/configuration.hpp"
#include "utilities/image_process.hpp"
#include "utilities/read_kitii_dataset.hpp"
#include "utilities/reconstructor.hpp"
#include "utilities/timer.hpp"
#include "utilities/visualizer.hpp"

namespace super_vio
{
enum class FrontEndStatus : std::uint8_t
{
  INITIALIZING  = 0,
  TRACKING_GOOD = 1,
  TRACKING_BAD  = 2,
  LOST          = 3,
};
inline std::ostream& operator<<( std::ostream& os, const FrontEndStatus& status )
{
  switch ( status )
  {
    case FrontEndStatus::INITIALIZING:
      os << "INITIALIZING";
      break;
    case FrontEndStatus::TRACKING_GOOD:
      os << "TRACKING_GOOD";
      break;
    case FrontEndStatus::TRACKING_BAD:
      os << "TRACKING_BAD";
      break;
    case FrontEndStatus::LOST:
      os << "LOST";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

class MapPoint;  // forward declaration
class Features;
class Frame;
class Extractor;
class Matcher;

class FrontEnd
{
private:
  std::size_t frame_id_ = 0;

  FrontEndStatus track_status_{ FrontEndStatus::INITIALIZING };

  utilities::Configuration config_;

  std::shared_ptr<Frame> last_frame_    = nullptr;
  std::shared_ptr<Frame> current_frame_ = nullptr;

  std::shared_ptr<Camera> left_camera_  = nullptr;
  std::shared_ptr<Camera> right_camera_ = nullptr;

  std::shared_ptr<Extractor> extractor_ = nullptr;
  std::shared_ptr<Matcher>   matcher_   = nullptr;


  // TODO: Add KeyFrame
  //   std::shared_ptr<KeyFrame> reference_keyframe_ = nullptr;
  // TODO: Add BackEnd Flag
  //   bool enable_backend_ = false;
  // TODO: Add IMUPreintegration
  //   bool enable_imu_preintegration_ = false;

public:
  FrontEnd() = delete;
  FrontEnd( const utilities::Configuration& config );
  ~FrontEnd();

  bool setStereoCamera( std::shared_ptr<Camera> left_camera, std::shared_ptr<Camera> right_camera );
  bool setExtractor( std::shared_ptr<Extractor> extractor );
  bool setMatcher( std::shared_ptr<Matcher> matcher );

  bool updateStereoImages( const cv::Mat& left_img, const cv::Mat& right_img, const double& timestamp );
  bool buildInitialMap();

  bool trackIterativeClosestPoint();
  bool trackBundleAdjust();
};


}  // namespace super_vio