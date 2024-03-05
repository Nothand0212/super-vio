#pragma once

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include "data/mapPoint.h"
#include "frame.h"

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

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < 1e-2 )
  {
    return true;
  }

  /// give up the bad solution
  return false;
}

inline bool triangulate( Eigen::Matrix<double, 3, 4> &pose_left, Eigen::Matrix<double, 3, 4> &pose_right,
                         Eigen::Vector2d &key_point_left, Eigen::Vector2d &key_point_right, Eigen::Vector3d &point_3d )
{
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row( 0 )        = key_point_left[ 0 ] * pose_left.row( 2 ) - pose_left.row( 0 );
  design_matrix.row( 1 )        = key_point_left[ 1 ] * pose_left.row( 2 ) - pose_left.row( 1 );
  design_matrix.row( 2 )        = key_point_right[ 0 ] * pose_right.row( 2 ) - pose_right.row( 0 );
  design_matrix.row( 3 )        = key_point_right[ 1 ] * pose_right.row( 2 ) - pose_right.row( 1 );

  auto svd = design_matrix.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV );
  point_3d = ( svd.matrixV().col( 3 ) / svd.matrixV()( 3, 3 ) ).head<3>();

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < 1e-2 )
  {
    return true;
  }

  return false;
}

inline bool triangulate( Eigen::Matrix<double, 3, 4> &pose_left, Eigen::Matrix<double, 3, 4> &pose_right,
                         cv::Point2f &key_point_left, cv::Point2f &key_point_right, Eigen::Vector3d &point_3d )
{
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row( 0 )        = key_point_left.x * pose_left.row( 2 ) - pose_left.row( 0 );
  design_matrix.row( 1 )        = key_point_left.y * pose_left.row( 2 ) - pose_left.row( 1 );
  design_matrix.row( 2 )        = key_point_right.x * pose_right.row( 2 ) - pose_right.row( 0 );
  design_matrix.row( 3 )        = key_point_right.y * pose_right.row( 2 ) - pose_right.row( 1 );

  auto svd = design_matrix.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV );
  point_3d = ( svd.matrixV().col( 3 ) / svd.matrixV()( 3, 3 ) ).head<3>();

  if ( svd.singularValues()[ 3 ] / svd.singularValues()[ 2 ] < 1e-2 )
  {
    return true;
  }

  return false;
}