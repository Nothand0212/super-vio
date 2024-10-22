#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "logger/mine_logger.hpp"
#include "nlohmann/json.hpp"
#include "super_vio/base_onnx_runner.hpp"
#include "super_vio/camera.hpp"
#include "super_vio/extractor.hpp"
#include "super_vio/frame.hpp"
#include "super_vio/map_point.hpp"
#include "super_vio/matcher.hpp"
#include "super_vio/pose_estimator_3d3d.hpp"
#include "utilities/accumulate_average.hpp"
#include "utilities/color.hpp"
#include "utilities/configuration.hpp"
#include "utilities/image_process.hpp"
#include "utilities/read_kitii_dataset.hpp"
#include "utilities/reconstructor.hpp"
#include "utilities/timer.hpp"
#include "utilities/visualizer.hpp"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>              VecX;
typedef Eigen::Matrix<double, 3, 4>                           Mat34;

struct StereoMatchingResult
{
  int index;

  cv::Point2f left_key_point;
  cv::Point2f right_key_point;

  Eigen::Vector3d point_3d;
};

std::vector<StereoMatchingResult> loadJson( const std::string& json_path )
{
  std::ifstream  ifs( json_path );
  nlohmann::json json_data;
  ifs >> json_data;
  ifs.close();

  std::vector<StereoMatchingResult> result;
  for ( auto& item : json_data )
  {
    // std::cout << item << std::endl;

    StereoMatchingResult r;
    r.index           = item[ "index" ];
    r.left_key_point  = cv::Point2f( item[ "keypoint" ][ "left" ][ "x" ], item[ "keypoint" ][ "left" ][ "y" ] );
    r.right_key_point = cv::Point2f( item[ "keypoint" ][ "right" ][ "x" ], item[ "keypoint" ][ "right" ][ "y" ] );
    r.point_3d        = Eigen::Vector3d( item[ "3DPoint" ][ 0 ], item[ "3DPoint" ][ 1 ], item[ "3DPoint" ][ 2 ] );
    result.push_back( r );
    std::cout << "\n";
    std::cout << "index: " << r.index << std::endl;
    std::cout << "left_key_point: " << r.left_key_point.x << ", " << r.left_key_point.y << std::endl;
    std::cout << "right_key_point: " << r.right_key_point.x << ", " << r.right_key_point.y << std::endl;
    std::cout << "point_3d: " << r.point_3d.x() << ", " << r.point_3d.y() << ", " << r.point_3d.z() << std::endl;
  }
  return result;
}

Eigen::Vector3d pixel2camera( const CameraParams& params, const cv::Point2f& point_cv, double depth = 1.0 )
{
  Eigen::Vector2d p_p = Eigen::Vector2d( point_cv.x, point_cv.y );

  return Eigen::Vector3d( ( p_p( 0, 0 ) - params.cx ) / params.fx * depth,
                          ( p_p( 1, 0 ) - params.cy ) / params.fy * depth,
                          depth );
}

Eigen::Vector3d pixel2camera( const CameraParams& params, const Eigen::Vector2d& p_p, double depth = 1.0 )
{
  return Eigen::Vector3d( ( p_p( 0, 0 ) - params.cx ) / params.fx * depth,
                          ( p_p( 1, 0 ) - params.cy ) / params.fy * depth,
                          depth );
}


inline bool triangulation( const std::vector<Sophus::SE3d>&   poses,
                           const std::vector<Eigen::Vector3d> points,
                           Eigen::Vector3d&                   pt_world )
{
  MatXX A( 2 * poses.size(), 4 );
  VecX  b( 2 * poses.size() );
  b.setZero();
  for ( size_t i = 0; i < poses.size(); ++i )
  {
    Mat34 m                       = poses[ i ].matrix3x4();
    A.block<1, 4>( 2 * i, 0 )     = points[ i ][ 0 ] * m.row( 2 ) - m.row( 0 );
    A.block<1, 4>( 2 * i + 1, 0 ) = points[ i ][ 1 ] * m.row( 2 ) - m.row( 1 );
  }
  auto svd = A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV );
  pt_world = ( svd.matrixV().col( 3 ) / svd.matrixV()( 3, 3 ) ).head<3>();

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < ERROR_THRESHOLD )
  {
    return true;
  }
  /// give up the bad solution
  return false;
}

int main( int argc, char** argv )
{
  std::string config_path = "/home/lin/Projects/ss_ws/src/super-vio/config/param.json";
  auto        json_data   = loadJson( "/home/lin/Projects/lk_ws/src/lk-vio/config/triangle.json" );

  utilities::Configuration cfg{};
  cfg.readConfigFile( config_path );

  // initialize camera
  std::shared_ptr<super_vio::Camera> camera_left_ptr = std::make_shared<super_vio::Camera>();
  camera_left_ptr->setParams( cfg.camera_matrix_left, cfg.distortion_coefficients_left );
  std::shared_ptr<super_vio::Camera> camera_right_ptr = std::make_shared<super_vio::Camera>();
  camera_right_ptr->setParams( cfg.camera_matrix_right, cfg.distortion_coefficients_right );


  float                       base_line_calculated = 386.1448 / 718.856;
  Eigen::Matrix<double, 3, 4> pose_left_tmp        = Eigen::Matrix<double, 3, 4>::Identity();
  Eigen::Matrix<double, 3, 4> pose_right_tmp       = Eigen::Matrix<double, 3, 4>::Identity();
  pose_right_tmp( 0, 3 )                           = -base_line_calculated;  // OpenCV is Left-Hand coordinate system, so the baseline is negative
  camera_left_ptr->setPose( pose_left_tmp );
  camera_right_ptr->setPose( pose_right_tmp );

  Eigen::Matrix<double, 3, 4> pose_left           = camera_left_ptr->getPoseMatrix();
  Eigen::Matrix<double, 3, 4> pose_right          = camera_right_ptr->getPoseMatrix();
  auto                        camera_params_left  = camera_left_ptr->getParams();
  auto                        camera_params_right = camera_right_ptr->getParams();

  std::cout << "\nCamera Left: \n"
            << camera_params_left.fx << ", " << camera_params_left.fy << ", " << camera_params_left.cx << ", " << camera_params_left.cy << std::endl;
  std::cout << "\nCamera Right: \n"
            << camera_params_right.fx << ", " << camera_params_right.fy << ", " << camera_params_right.cx << ", " << camera_params_right.cy << std::endl;

  std::cout << "\nPose Left: \n"
            << camera_left_ptr->getPose().matrix() << std::endl;
  std::cout << "\nPose Right: \n"
            << camera_right_ptr->getPose().matrix() << std::endl;

  std::vector<Sophus::SE3d> poses;
  poses.push_back( camera_left_ptr->getPose() );
  poses.push_back( camera_right_ptr->getPose() );

  // auto cv_point2f_to_vec2 = []( cv::Point2f& pt ) { return Eigen::Vector2d( pt.x, pt.y ); };

  for ( const auto& item : json_data )
  {
    Eigen::Vector3d point_3d_calculated;
    // utilities::compute3DPoint( camera_params_left, base_line_calculated, item.left_key_point, item.right_key_point, point_3d_calculated );
    std::vector<Eigen::Vector3d> points;
    points.push_back( pixel2camera( camera_params_left, item.left_key_point ) );
    points.push_back( pixel2camera( camera_params_right, item.right_key_point ) );

    triangulation( poses, points, point_3d_calculated );
    std::cout << "Index: " << item.index << std::endl;
    std::cout << "3D Point Calculated: " << point_3d_calculated.x() << ", " << point_3d_calculated.y() << ", " << point_3d_calculated.z() << std::endl;
    std::cout << "3D Point Ground Truth: " << item.point_3d.x() << ", " << item.point_3d.y() << ", " << item.point_3d.z() << std::endl;
  }
}