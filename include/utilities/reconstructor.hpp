#pragma once
#define SOPHUS_USE_BASIC_LOGGING
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

// #include "data/map_point.hpp"
#include "super_vio/camera.hpp"
#include "super_vio/frame.hpp"
#include "super_vio/map_point.hpp"

namespace utilities
{
#define ERROR_THRESHOLD 1e-1
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

bool compute3DPoint( const super_vio::CameraParams &camera_params_left, const float &base_line, const cv::Point2f &pixel_left, const cv::Point2f &pixel_right, Eigen::Vector3d &point_3d )
{
  // std::cout << "Disparity calculation...\n";
  // calculate disparity
  float disparity = pixel_left.x - pixel_right.x;

  // calculate 3D point
  if ( disparity <= 0.1f )
  {
    return false;
  }

  float depth = base_line * camera_params_left.fx / disparity;

  float x = ( ( pixel_left.x - camera_params_left.cx ) * depth ) / camera_params_left.fx;
  float y = ( ( pixel_left.y - camera_params_left.cy ) * depth ) / camera_params_left.fy;
  float z = depth;


  point_3d = Eigen::Vector3d( x, y, z );

  return true;
}

inline void eigenMatrixToCvMat( const Eigen::Matrix<double, 3, 4> &eigen_mat, cv::Mat &cv_mat )
{
  // 创建一个 3 行 4 列的 cv::Mat，并确保类型为 float
  cv_mat.create( 3, 4, CV_32F );
  for ( int i = 0; i < 3; ++i )
  {
    for ( int j = 0; j < 4; ++j )
    {
      cv_mat.at<float>( i, j ) = static_cast<float>( eigen_mat( i, j ) );
    }
  }
}

void compute3DPoints( const super_vio::CameraParams &camera_params_left, const super_vio::CameraParams &camera_params_right,
                      const Eigen::Matrix<double, 3, 4> &pose_left, const Eigen::Matrix<double, 3, 4> &pose_right,
                      const std::vector<cv::Point2f> &pixel_left, const std::vector<cv::Point2f> &pixel_right, std::vector<Eigen::Vector3d> &point_3d )
{
  std::cout << "Transforming points to camera coordinates...\n";
  std::vector<cv::Point2f> pt_cam_left, pt_cam_right;
  for ( size_t i = 0; i < pixel_left.size(); ++i )
  {
    cv::Point2f pt_cam_left_i  = pixelToCamera( pixel_left[ i ], camera_params_left );
    cv::Point2f pt_cam_right_i = pixelToCamera( pixel_right[ i ], camera_params_right );
    pt_cam_left.push_back( pt_cam_left_i );
    pt_cam_right.push_back( pt_cam_right_i );
  }

  std::cout << "Transforming Eigen to CV Mat...\n";
  cv::Mat T_left, T_right;
  // eigenMatrixToCvMat( pose_left, T_left );
  // eigenMatrixToCvMat( pose_right, T_right );
  T_left  = ( cv::Mat_<float>( 3, 4 ) << pose_left( 0, 0 ), pose_left( 0, 1 ), pose_left( 0, 2 ), pose_left( 0, 3 ),
             pose_left( 1, 0 ), pose_left( 1, 1 ), pose_left( 1, 2 ), pose_left( 1, 3 ),
             pose_left( 2, 0 ), pose_left( 2, 1 ), pose_left( 2, 2 ), pose_left( 2, 3 ) );
  T_right = ( cv::Mat_<float>( 3, 4 ) << pose_right( 0, 0 ), pose_right( 0, 1 ), pose_right( 0, 2 ), pose_right( 0, 3 ),
              pose_right( 1, 0 ), pose_right( 1, 1 ), pose_right( 1, 2 ), pose_right( 1, 3 ),
              pose_right( 2, 0 ), pose_right( 2, 1 ), pose_right( 2, 2 ), pose_right( 2, 3 ) );

  std::cout << "Triangulating points...\n";
  cv::Mat points_4d;
  cv::triangulatePoints( T_left, T_right, pt_cam_left, pt_cam_right, points_4d );

  std::cout << "Extracting 3D points...\n";
  for ( int i = 0; i < points_4d.cols; i++ )
  {
    cv::Mat x = points_4d.col( i );
    x /= x.at<float>( 3, 0 );
    Eigen::Vector3d point_3d_i( x.at<float>( 0, 0 ), x.at<float>( 1, 0 ), x.at<float>( 2, 0 ) );
    point_3d.push_back( point_3d_i );
  }
}

inline std::vector<cv::Point2f> projection3DPointToPixel( const std::vector<cv::Point3f> &points_3d, const super_vio::CameraParams &camera_params )
{
  std::vector<cv::Point2f> points2D;
  points2D.reserve( points_3d.size() );

  cv::Mat K = ( cv::Mat_<float>( 3, 3 ) << camera_params.fx, 0, camera_params.cx, 0, camera_params.fy, camera_params.cy, 0, 0, 1 );
  std::cout << "K:\n"
            << K << std::endl;
  cv::Mat distortion_coefficients = ( cv::Mat_<float>( 1, 5 ) << camera_params.k1, camera_params.k2, camera_params.p1, camera_params.p2, camera_params.k3 );
  std::cout << "distortion_coefficients:\n"
            << distortion_coefficients << std::endl;


  for ( size_t i = 0; i < points_3d.size(); ++i )
  {
    cv::Mat pointHomogeneous = ( cv::Mat_<float>( 3, 1 ) << points_3d[ i ].x, points_3d[ i ].y, points_3d[ i ].z );
    cv::Mat pointProjected   = K * pointHomogeneous;
    if ( pointProjected.at<float>( 2 ) != 0 )
    {  // 避免除以0
      cv::Point2f point2D(
          pointProjected.at<float>( 0 ) / pointProjected.at<float>( 2 ),
          pointProjected.at<float>( 1 ) / pointProjected.at<float>( 2 ) );
      points2D.push_back( point2D );
    }
  }

  return points2D;
}


}  // namespace utilities