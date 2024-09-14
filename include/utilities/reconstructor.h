#pragma once
#define SOPHUS_USE_BASIC_LOGGING
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

// #include "data/mapPoint.h"
#include "super_vio/camera.hpp"
#include "super_vio/frame.h"
#include "super_vio/mapPoint.h"

namespace utilities
{
#define ERROR_THRESHOLD 1e-2
/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulate( const std::vector<Sophus::SE3d> &  poses,
                         const std::vector<Eigen::Vector3d> points,
                         Eigen::Vector3d &                  pt_world )
{
  Eigen::Matrix<double, -1, -1> A( 2 * poses.size(), 4 );
  Eigen::Matrix<double, -1, 1>  b( 2 * poses.size() );

  b( 2 * poses.size() );
  b.setZero();
  for ( size_t i = 0; i < poses.size(); ++i )
  {
    Eigen::Matrix<double, 3, 4> m = poses[ i ].matrix3x4();
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


inline bool triangulate( const std::vector<Sophus::SE3d> &poses,
                         const std::vector<cv::Point2f>   points,
                         Eigen::Vector3d &                pt_world )
{
  Eigen::Matrix<double, -1, -1> A( 2 * poses.size(), 4 );
  Eigen::Matrix<double, -1, 1>  b( 2 * poses.size() );

  b( 2 * poses.size() );
  b.setZero();
  for ( size_t i = 0; i < poses.size(); ++i )
  {
    Eigen::Matrix<double, 3, 4> m = poses[ i ].matrix3x4();
    A.block<1, 4>( 2 * i, 0 )     = points[ i ].x * m.row( 2 ) - m.row( 0 );
    A.block<1, 4>( 2 * i + 1, 0 ) = points[ i ].y * m.row( 2 ) - m.row( 1 );
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


inline bool triangulate( const Eigen::Matrix<double, 3, 4> &pose_left, const Eigen::Matrix<double, 3, 4> &pose_right,
                         const Eigen::Vector2d &key_point_left, const Eigen::Vector2d &key_point_right, Eigen::Vector3d &point_3d )
{
  Eigen::MatrixXd design_matrix = Eigen::MatrixXd::Zero( 4, 4 );
  design_matrix.row( 0 )        = key_point_left[ 0 ] * pose_left.row( 2 ) - pose_left.row( 0 );
  design_matrix.row( 1 )        = key_point_left[ 1 ] * pose_left.row( 2 ) - pose_left.row( 1 );
  design_matrix.row( 2 )        = key_point_right[ 0 ] * pose_right.row( 2 ) - pose_right.row( 0 );
  design_matrix.row( 3 )        = key_point_right[ 1 ] * pose_right.row( 2 ) - pose_right.row( 1 );

  auto svd = design_matrix.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV );
  point_3d = ( svd.matrixV().col( 3 ) / svd.matrixV()( 3, 3 ) ).head<3>();

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < ERROR_THRESHOLD )
  {
    return true;
  }

  return false;
}

inline bool triangulate( const Eigen::Matrix<double, 3, 4> &pose_left, const Eigen::Matrix<double, 3, 4> &pose_right,
                         const cv::Point2f &key_point_left, const cv::Point2f &key_point_right, Eigen::Vector3d &point_3d )
{
  Eigen::MatrixXd design_matrix = Eigen::MatrixXd::Zero( 4, 4 );
  design_matrix.row( 0 )        = key_point_left.x * pose_left.row( 2 ) - pose_left.row( 0 );
  design_matrix.row( 1 )        = key_point_left.y * pose_left.row( 2 ) - pose_left.row( 1 );
  design_matrix.row( 2 )        = key_point_right.x * pose_right.row( 2 ) - pose_right.row( 0 );
  design_matrix.row( 3 )        = key_point_right.y * pose_right.row( 2 ) - pose_right.row( 1 );

  Eigen::JacobiSVD<Eigen::MatrixXd> svd( design_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
  point_3d = ( svd.matrixV().col( 3 ) / svd.matrixV()( 3, 3 ) ).head<3>();

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < ERROR_THRESHOLD )
  {
    return true;
  }

  return false;
}


cv::Point2f pixelToCamera( const cv::Point2d &p, const cv::Mat &K )
{
  cv::Point2f pt_cam;

  pt_cam.x = ( p.x - K.at<float>( 0, 2 ) ) / K.at<float>( 0, 0 );
  pt_cam.y = ( p.y - K.at<float>( 1, 2 ) ) / K.at<float>( 1, 1 );

  return pt_cam;
}
cv::Point2f pixelToCamera( const cv::Point2d &p, const super_vio::CameraParams &camera_params )
{
  cv::Point2f pt_cam;

  pt_cam.x = ( p.x - camera_params.cx ) / camera_params.fx;
  pt_cam.y = ( p.y - camera_params.cy ) / camera_params.fy;

  return pt_cam;
}


bool compute3DPoint( const cv::Mat &K_left, const cv::Mat &K_right,
                     const Eigen::Matrix<double, 3, 4> &pose_left, const Eigen::Matrix<double, 3, 4> &pose_right,
                     const cv::Point2f &pixel_left, const cv::Point2f &pixel_right, Eigen::Vector3d &point_3d )
{
  // Convert pixel coordinates to camera coordinates
  cv::Point2f pt_cam_left  = pixelToCamera( pixel_left, K_left );
  cv::Point2f pt_cam_right = pixelToCamera( pixel_right, K_right );

  // Triangulate to compute 3D point
  bool valid = triangulate( pose_left, pose_right, pt_cam_left, pt_cam_right, point_3d );

  return valid;
}

bool compute3DPoint( const super_vio::CameraParams &camera_params_left, const super_vio::CameraParams &camera_params_right,
                     const Eigen::Matrix<double, 3, 4> &pose_left, const Eigen::Matrix<double, 3, 4> &pose_right,
                     const cv::Point2f &pixel_left, const cv::Point2f &pixel_right, Eigen::Vector3d &point_3d )
{
  // Convert pixel coordinates to camera coordinates
  cv::Point2f pt_cam_left  = pixelToCamera( pixel_left, camera_params_left );
  cv::Point2f pt_cam_right = pixelToCamera( pixel_right, camera_params_right );

  // Triangulate to compute 3D point
  bool valid = triangulate( pose_left, pose_right, pt_cam_left, pt_cam_right, point_3d );

  return valid;
}


bool compute3DPoint( const cv::Mat &K_left, const float &base_line, const cv::Point2f &pixel_left, const cv::Point2f &pixel_right, Eigen::Vector3d &point_3d )
{
  // calculate disparity
  float disparity = pixel_left.x - pixel_right.x;

  // calculate 3D point
  if ( disparity <= 0.1f )
  {
    return false;
  }

  float depth = base_line * K_left.at<float>( 0, 0 ) / disparity;

  float x = ( ( pixel_left.x - K_left.at<float>( 0, 2 ) ) * depth ) / K_left.at<float>( 0, 0 );
  float y = ( ( pixel_left.y - K_left.at<float>( 1, 2 ) ) * depth ) / K_left.at<float>( 1, 1 );
  float z = depth;


  point_3d = Eigen::Vector3d( x, y, z );

  return true;
}
}  // namespace utilities