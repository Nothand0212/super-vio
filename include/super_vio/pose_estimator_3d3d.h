#pragma once

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "logger/logger.h"
#include "super_vio/extractor.h"
#include "super_vio/matcher.h"
#include "utilities/accumulate_average.h"
#include "utilities/configuration.h"
#include "utilities/visualizer.h"

namespace super_vio
{
#define DISTANCE_THRES 3.0f

cv::Point3f calculateCentroid( const std::vector<cv::Point3f>& points )
{
  // std::ostringstream oss;

  int         size = points.size();
  cv::Point3f centroid( 0, 0, 0 );
  // oss << "\nSie: " << size;
  // oss << "\nCentroid: " << centroid.x << " " << centroid.y << " " << centroid.z;
  for ( const auto& point : points )
  {
    // oss << "\nCurrent Point: " << point.x << " " << point.y << " " << point.z;
    centroid += point;
  }
  centroid.x /= size;
  centroid.y /= size;
  centroid.z /= size;
  // oss << "\nFinal Centroid: " << centroid.x << " " << centroid.y << " " << centroid.z;
  // INFO( super_vio::logger, oss.str() );
  return centroid;
};

bool isValid( const cv::Point3f& point )
{
  if ( std::isnan( point.x ) || std::isnan( point.y ) || std::isnan( point.z ) ||
       std::isinf( point.x ) || std::isinf( point.y ) || std::isinf( point.z ) )
  {
    return false;
  }

  if ( std::abs( point.x ) < 1e-4 && std::abs( point.y ) < 1e-4 && std::abs( point.z ) < 1e-4 )
  {
    return false;
  }

  if ( std::abs( point.x ) > 1e3 || std::abs( point.y ) > 1e3 || std::abs( point.z ) > 1e3 )
  {
    return false;
  }

  return true;
};

bool isValid( const cv::Point3f& point_a, const cv::Point3f& point_b )
{
  bool condition_1 = isValid( point_a ) && isValid( point_b );

  bool condition_2 = cv::norm( point_a - point_b ) < DISTANCE_THRES;

  return condition_1 && condition_2;
};


/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override
  {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl( const double* update ) override
  {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[ 0 ], update[ 1 ], update[ 2 ], update[ 3 ], update[ 4 ], update[ 5 ];
    _estimate = Sophus::SE3d::exp( update_eigen ) * _estimate;
  }

  virtual bool read( istream& in ) override {}

  virtual bool write( ostream& out ) const override {}
};

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point( point ) {}

  virtual void computeError() override
  {
    const VertexPose* pose = static_cast<const VertexPose*>( _vertices[ 0 ] );
    _error                 = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override
  {
    VertexPose*     pose                 = static_cast<VertexPose*>( _vertices[ 0 ] );
    Sophus::SE3d    T                    = pose->estimate();
    Eigen::Vector3d xyz_trans            = T * _point;
    _jacobianOplusXi.block<3, 3>( 0, 0 ) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>( 0, 3 ) = Sophus::SO3d::hat( xyz_trans );
  }

  bool read( istream& in ) {}

  bool write( ostream& out ) const {}

protected:
  Eigen::Vector3d _point;
};


class PoseEstimator3D3D
{
private:
  float scale_;

  cv::Mat                  last_image_;
  std::vector<cv::Point2f> last_keypoints_;
  std::vector<cv::Point3f> last_points3d_;
  cv::Mat                  last_descriptors_;

  cv::Mat                  current_image_;
  std::vector<cv::Point2f> current_keypoints_;
  std::vector<cv::Point3f> current_points3d_;
  cv::Mat                  current_descriptors_;


  std::shared_ptr<utilities::Configuration> config_sptr_;
  std::shared_ptr<Matcher>                  matcher_sptr_;

  bool is_initialized_{ false };

public:
  PoseEstimator3D3D() = delete;
  PoseEstimator3D3D( std::shared_ptr<Matcher> matcher_sptr, std::shared_ptr<utilities::Configuration> config_sptr, float scale = 1.0f );
  ~PoseEstimator3D3D();

  void                                               setScale( float scale );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> setData( const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, const std::vector<cv::Point3f>& points3d, const cv::Mat& descriptors );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       optimizePose();
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       getPoseSVD( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       getPoseG2O( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current );

  bool getInitializedFlag() const;
};

PoseEstimator3D3D::PoseEstimator3D3D( std::shared_ptr<Matcher> matcher_sptr, std::shared_ptr<utilities::Configuration> config_sptr, float scale )
{
  matcher_sptr_ = matcher_sptr;
  config_sptr_  = config_sptr;
  scale_        = scale;


  is_initialized_ = false;
  INFO( super_vio::logger, "PoseEstimator3D3D initialized" );
}

PoseEstimator3D3D::~PoseEstimator3D3D()
{
  // TODO: release resources
  INFO( super_vio::logger, "PoseEstimator3D3D destroyed" );
}

void PoseEstimator3D3D::setScale( float scale )
{
  this->scale_ = scale;
}

bool PoseEstimator3D3D::getInitializedFlag() const
{
  return this->is_initialized_;
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> PoseEstimator3D3D::setData( const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, const std::vector<cv::Point3f>& points3d, const cv::Mat& descriptors )
{
  INFO( super_vio::logger, "Received new frame, keypoints: {0}, points3d: {1}, descriptors: {2}", keypoints.size(), points3d.size(), descriptors.rows );
  if ( this->is_initialized_ == true )
  {
    this->last_image_       = this->current_image_;
    this->last_keypoints_   = this->current_keypoints_;
    this->last_points3d_    = this->current_points3d_;
    this->last_descriptors_ = this->current_descriptors_;

    this->current_image_       = img;
    this->current_keypoints_   = keypoints;
    this->current_points3d_    = points3d;
    this->current_descriptors_ = descriptors;

    auto [ rotation, translation ] = this->optimizePose();
    return std::make_tuple( rotation, translation, true );
  }
  else
  {
    INFO( super_vio::logger, "Receiving the first frame" );

    this->current_image_       = img;
    this->current_keypoints_   = keypoints;
    this->current_points3d_    = points3d;
    this->current_descriptors_ = descriptors;

    this->is_initialized_ = true;
    return std::make_tuple( Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), false );
  }
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator3D3D::optimizePose()
{
  // 1. Match the keypoints between the last and current frame
  auto matches_set = this->matcher_sptr_->inferenceDescriptorPair( *this->config_sptr_,
                                                                   this->last_keypoints_, this->current_keypoints_,
                                                                   this->last_descriptors_, this->current_descriptors_ );

  // 2. Extract the 3D points from the matches

  std::vector<cv::Point2f> key_points_transformed_src = getKeyPointsInOriginalImage( this->last_keypoints_, this->scale_ );
  std::vector<cv::Point2f> key_points_transformed_dst = getKeyPointsInOriginalImage( this->current_keypoints_, this->scale_ );

  std::vector<cv::Point3f> matches_3d_src;
  std::vector<cv::Point3f> matches_3d_dst;

  std::vector<cv::Point2f> matches_2d_src;
  std::vector<cv::Point2f> matches_2d_dst;
  std::ostringstream       oss;
  for ( const auto& match : matches_set )
  {
    if ( isValid( this->last_points3d_[ match.first ], this->current_points3d_[ match.second ] ) )
    {
      matches_3d_src.emplace_back( this->last_points3d_[ match.first ] );
      matches_3d_dst.emplace_back( this->current_points3d_[ match.second ] );
      oss << "\n\nMatch: " << match.first << " " << match.second;
      oss << "\nLast";
      oss << "\nKeyPoint: " << key_points_transformed_src[ match.first ].x << " " << key_points_transformed_src[ match.first ].y;
      oss << "\nPoint3D: " << this->last_points3d_[ match.first ].x << " " << this->last_points3d_[ match.first ].y << " " << this->last_points3d_[ match.first ].z;
      oss << "\nCurrent";
      oss << "\nKeyPoint: " << key_points_transformed_dst[ match.second ].x << " " << key_points_transformed_dst[ match.second ].y;
      oss << "\nPoint3D: " << this->current_points3d_[ match.second ].x << " " << this->current_points3d_[ match.second ].y << " " << this->current_points3d_[ match.second ].z;
    }

    matches_2d_src.emplace_back( key_points_transformed_src[ match.first ] );
    matches_2d_dst.emplace_back( key_points_transformed_dst[ match.second ] );
  }
  INFO( super_vio::logger, oss.str() );
  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_2d_pair = std::make_pair( matches_2d_src, matches_2d_dst );


  INFO( super_vio::logger, "Matches: {0}, key points: {1}", matches_2d_pair.first.size(), key_points_transformed_src.size() );
  INFO( super_vio::logger, "Last image: [{0} x {1}], Current image: [{2} x {3}]", this->last_image_.cols, this->last_image_.rows, this->current_image_.cols, this->current_image_.rows );
  auto img = visualizeMatches( this->last_image_, this->current_image_, matches_2d_pair, key_points_transformed_src, key_points_transformed_dst );

  cv::imshow( "Last-Current Matches", img );
  cv::waitKey( 0 );

  // 3. Estimate the pose using the 3D-3D correspondences
  // return this->getPoseSVD( matches_3d_src, matches_3d_dst );
  return this->getPoseG2O( matches_3d_src, matches_3d_dst );
}


std::tuple<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator3D3D::getPoseSVD( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current )
{
  std::ostringstream oss;
  // 1. Calculate the centroids of the two sets of points
  std::size_t num_points = points_last.size();

  cv::Point3f centroid_last    = calculateCentroid( points_last );
  cv::Point3f centroid_current = calculateCentroid( points_current );
  oss << "\nLast Centroid = " << centroid_last.x << " " << centroid_last.y << " " << centroid_last.z;
  oss << "\nCurrent Centroid = " << centroid_current.x << " " << centroid_current.y << " " << centroid_current.z;
  INFO( super_vio::logger, oss.str() );
  oss.str( "" );

  // 2. Remove the centroid from the points
  std::vector<cv::Point3f> points_last_centered( num_points );
  std::vector<cv::Point3f> points_current_centered( num_points );
  for ( std::size_t i = 0; i < num_points; ++i )
  {
    points_last_centered[ i ]    = points_last[ i ] - centroid_last;
    points_current_centered[ i ] = points_current[ i ] - centroid_current;
  }


  // 3. Compute points_last_centered * points_current_centered^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for ( std::size_t i = 0; i < num_points; ++i )
  {
    W += Eigen::Vector3d( points_last_centered[ i ].x, points_last_centered[ i ].y, points_last_centered[ i ].z ) * Eigen::Vector3d( points_current_centered[ i ].x, points_current_centered[ i ].y, points_current_centered[ i ].z ).transpose();
  }

  oss << "\nW = ";
  oss << "\n\t" << W( 0, 0 ) << " " << W( 0, 1 ) << " " << W( 0, 2 );
  oss << "\n\t" << W( 1, 0 ) << " " << W( 1, 1 ) << " " << W( 1, 2 );
  oss << "\n\t" << W( 2, 0 ) << " " << W( 2, 1 ) << " " << W( 2, 2 );


  // 4. Compute SVD of W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd( W, Eigen::ComputeFullU | Eigen::ComputeFullV );

  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  oss << "\nU = ";
  oss << "\n\t" << U( 0, 0 ) << " " << U( 0, 1 ) << " " << U( 0, 2 );
  oss << "\n\t" << U( 1, 0 ) << " " << U( 1, 1 ) << " " << U( 1, 2 );
  oss << "\n\t" << U( 2, 0 ) << " " << U( 2, 1 ) << " " << U( 2, 2 );

  oss << "\nV = ";
  oss << "\n\t" << V( 0, 0 ) << " " << V( 0, 1 ) << " " << V( 0, 2 );
  oss << "\n\t" << V( 1, 0 ) << " " << V( 1, 1 ) << " " << V( 1, 2 );
  oss << "\n\t" << V( 2, 0 ) << " " << V( 2, 1 ) << " " << V( 2, 2 );

  Eigen::Matrix3d rotation = U * ( V.transpose() );
  if ( rotation.determinant() < 0 )
  {
    rotation = -rotation;
  }
  // transform to euler angle(roll, pitch, yaw)
  Eigen::Vector3d euler_angles = rotation.eulerAngles( 0, 1, 2 );


  Eigen::Vector3d translation = Eigen::Vector3d( centroid_last.x, centroid_last.y, centroid_last.z ) - rotation * Eigen::Vector3d( centroid_current.x, centroid_current.y, centroid_current.z );


  oss << "\nRotation = ";
  oss << "\n\t" << rotation( 0, 0 ) << " " << rotation( 0, 1 ) << " " << rotation( 0, 2 );
  oss << "\n\t" << rotation( 1, 0 ) << " " << rotation( 1, 1 ) << " " << rotation( 1, 2 );
  oss << "\n\t" << rotation( 2, 0 ) << " " << rotation( 2, 1 ) << " " << rotation( 2, 2 );
  oss << "\nEuler angles = " << euler_angles( 0 ) << " " << euler_angles( 1 ) << " " << euler_angles( 2 );
  oss << "\nTranslation = " << translation( 0 ) << " " << translation( 1 ) << " " << translation( 2 );
  INFO( super_vio::logger, oss.str() );

  return std::make_tuple( rotation, translation );
}


std::tuple<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator3D3D::getPoseG2O( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current )
{
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolverX                                       BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto                 solver = new g2o::OptimizationAlgorithmDogleg( std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() ) );
  g2o::SparseOptimizer optimizer;    // 图模型
  optimizer.setAlgorithm( solver );  // 设置求解器
  optimizer.setVerbose( true );      // 打开调试输出

  // vertex
  VertexPose* pose = new VertexPose();  // camera pose
  pose->setId( 0 );
  pose->setEstimate( Sophus::SE3d() );
  optimizer.addVertex( pose );

  // edges
  for ( size_t i = 0; i < points_last.size(); i++ )
  {
    EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly( Eigen::Vector3d( points_current[ i ].x, points_current[ i ].y, points_current[ i ].z ) );
    edge->setVertex( 0, pose );
    edge->setMeasurement( Eigen::Vector3d( points_last[ i ].x, points_last[ i ].y, points_last[ i ].z ) );
    edge->setInformation( Eigen::Matrix3d::Identity() );
    optimizer.addEdge( edge );
  }

  optimizer.initializeOptimization();
  optimizer.optimize( 100 );

  Eigen::Matrix3d rotation     = pose->estimate().rotationMatrix();
  Eigen::Vector3d translation  = pose->estimate().translation();
  Eigen::Vector3d euler_angles = rotation.eulerAngles( 0, 1, 2 );

  std::ostringstream oss;
  oss << "\nRotation = ";
  oss << "\n\t" << rotation( 0, 0 ) << " " << rotation( 0, 1 ) << " " << rotation( 0, 2 );
  oss << "\n\t" << rotation( 1, 0 ) << " " << rotation( 1, 1 ) << " " << rotation( 1, 2 );
  oss << "\n\t" << rotation( 2, 0 ) << " " << rotation( 2, 1 ) << " " << rotation( 2, 2 );
  oss << "\nEuler angles = " << euler_angles( 0 ) << " " << euler_angles( 1 ) << " " << euler_angles( 2 );
  oss << "\nTranslation = " << translation( 0 ) << " " << translation( 1 ) << " " << translation( 2 );
  INFO( super_vio::logger, oss.str() );
  return std::make_tuple( rotation, translation );
}

}  // namespace super_vio