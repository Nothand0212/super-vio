#pragma once

// json
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

// pcl
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

// ros
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <sophus/se3.hpp>

#include "logger/mine_logger.hpp"
#include "utilities/color.hpp"
#include "utilities/configuration.hpp"
#include "utilities/timer.hpp"

#define PATH true

namespace super_vio
{
class ROSTool
{
private:
  ros::NodeHandle nh_;

  std::string current_cloud_topic_;
  std::string global_cloud_topic_;
  std::string current_pose_topic_;
  std::string path_topic_;

  ros::Publisher pub_current_cloud_;
  ros::Publisher pub_global_cloud_;
  ros::Publisher pub_pose_;
  ros::Publisher pub_path_;

  nav_msgs::Path global_path_;
  nav_msgs::Path key_frame_path_;

public:
  ROSTool( const ros::NodeHandle& nh, const std::string& path );
  ~ROSTool() = default;

  void loadParamJson( const std::string& path );
  void publishPointCloud( const std::vector<Eigen::Vector3d>& points, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, double time_stamp = -1.0 );
  void publishPose( const Sophus::SE3d& pose, double time_stamp = -1.0 );
};


}  // namespace super_vio