#include "super_vio/ros_tool.hpp"


namespace super_vio
{
ROSTool::ROSTool( const ros::NodeHandle& nh, const std::string& path ) : nh_( nh )
{
  this->loadParamJson( path );

  this->pub_current_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>( this->current_cloud_topic_, 10 );
  this->pub_global_cloud_  = nh_.advertise<sensor_msgs::PointCloud2>( this->global_cloud_topic_, 10 );
  this->pub_pose_          = nh_.advertise<geometry_msgs::PoseStamped>( this->current_pose_topic_, 10 );
  this->pub_path_          = nh_.advertise<nav_msgs::Path>( this->path_topic_, 10 );

  image_transport::ImageTransport it( nh_ );
  this->pub_image_ = it.advertise( this->image_topic_, 1 );
}

void ROSTool::loadParamJson( const std::string& path )
{
  nlohmann::json json_data;
  std::ifstream  ifs( path );
  ifs >> json_data;
  ifs.close();

  this->current_cloud_topic_ = json_data[ "current_cloud_topic" ];
  this->global_cloud_topic_  = json_data[ "global_cloud_topic" ];
  this->current_pose_topic_  = json_data[ "current_pose_topic" ];
  this->path_topic_          = json_data[ "path_topic" ];
  this->image_topic_         = json_data[ "image_topic" ];

  INFO( super_vio::logger, "Current cloud topic: {0}", this->current_cloud_topic_ );
  INFO( super_vio::logger, "Global cloud topic:  {0}", this->global_cloud_topic_ );
  INFO( super_vio::logger, "Current pose topic:  {0}", this->current_pose_topic_ );
  INFO( super_vio::logger, "Path topic:          {0}", this->path_topic_ );
  INFO( super_vio::logger, "Image topic:         {0}", this->image_topic_ );
}

void ROSTool::publishCurrentPointCloud( const std::vector<Eigen::Vector3d>& points, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, double time_stamp )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  for ( const auto& point : points )
  {
    Eigen::Vector3d transformed_point = rotation * point + translation;
    cloud->points.push_back( pcl::PointXYZ( transformed_point[ 0 ], transformed_point[ 1 ], transformed_point[ 2 ] ) );
  }
  cloud->width    = cloud->points.size();
  cloud->height   = 1;
  cloud->is_dense = true;

  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg( *cloud, output );
  output.header.frame_id = "camera_link";

  if ( time_stamp < 0.0 )
  {
    output.header.stamp = ros::Time::now();
  }
  else
  {
    output.header.stamp = ros::Time( time_stamp );
  }

  this->pub_current_cloud_.publish( output );
}


void ROSTool::publishGlobalPointCloud( const std::vector<std::vector<Eigen::Vector3d>>& points_buffer, const std::vector<Sophus::SE3d>& pose_buffer, double time_stamp )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );
  assert( points_buffer.size() == pose_buffer.size() );

  for ( std::size_t i = 0; i < pose_buffer.size(); i++ )
  {
    for ( const auto& point : points_buffer[ i ] )
    {
      Eigen::Vector3d transformed_point = pose_buffer[ i ] * point;
      cloud->points.push_back( pcl::PointXYZ( transformed_point[ 0 ], transformed_point[ 1 ], transformed_point[ 2 ] ) );
    }
  }
  cloud->width    = cloud->points.size();
  cloud->height   = 1;
  cloud->is_dense = true;

  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg( *cloud, output );
  output.header.frame_id = "camera_link";

  if ( time_stamp < 0.0 )
  {
    output.header.stamp = ros::Time::now();
  }
  else
  {
    output.header.stamp = ros::Time( time_stamp );
  }

  this->pub_global_cloud_.publish( output );
}

void ROSTool::publishPose( const Sophus::SE3d& pose, double time_stamp )
{
  geometry_msgs::PoseStamped ros_pose;
  if ( time_stamp < 0.0 )
  {
    ros_pose.header.stamp = ros::Time::now();
  }
  else
  {
    ros_pose.header.stamp = ros::Time( time_stamp );
  }

  ros_pose.header.frame_id = "camera_link";

  Eigen::Matrix3d    rotation = pose.rotationMatrix();
  Eigen::Quaterniond quat( rotation );
  Eigen::Vector3d    translation = pose.translation();

  ros_pose.pose.position.x    = translation[ 0 ];
  ros_pose.pose.position.y    = translation[ 1 ];
  ros_pose.pose.position.z    = translation[ 2 ];
  ros_pose.pose.orientation.x = quat.x();
  ros_pose.pose.orientation.y = quat.y();
  ros_pose.pose.orientation.z = quat.z();
  ros_pose.pose.orientation.w = quat.w();

  this->pub_pose_.publish( ros_pose );

#if PATH
  this->global_path_.header = ros_pose.header;
  this->global_path_.poses.push_back( ros_pose );
  this->pub_path_.publish( this->global_path_ );
#endif
}

void ROSTool::publishImage( const cv::Mat& image )
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage( std_msgs::Header(), "bgr8", image ).toImageMsg();

  this->pub_image_.publish( msg );
}


}  // namespace super_vio