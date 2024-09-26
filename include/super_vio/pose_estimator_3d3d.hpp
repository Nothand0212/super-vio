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
#include "utilities/timer.h"
#include "utilities/visualizer.h"
namespace super_vio
{
#define DISTANCE_THRES 3.0f

using BlockSolverType  = g2o::BlockSolverX;
using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;


cv::Point3f calculateCentroid( const std::vector<cv::Point3f>& points );

bool isValid( const cv::Point3f& point );

bool isValid( const cv::Point3f& point_a, const cv::Point3f& point_b );

cv::Point3f transFromEigen( const Eigen::Vector3d& vec );

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

  virtual bool read( istream& in ) override
  {
    return false;
  }

  virtual bool write( ostream& out ) const override
  {
    return false;
  }
};

/// g2o edge
class EdgeProjectXYZPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZPoseOnly( const Eigen::Vector3d& point ) : _point( point ) {}

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

  // g2o related
  g2o::SparseOptimizer optimizer_;  // 图模型
  Sophus::SE3d         last_pose_;

  // for debug
  utilities::Timer timer_;
  cv::Mat          debug_image_;

public:
  PoseEstimator3D3D() = delete;
  PoseEstimator3D3D( std::shared_ptr<Matcher> matcher_sptr, std::shared_ptr<utilities::Configuration> config_sptr, float scale = 1.0f );
  ~PoseEstimator3D3D();

  void                                               setScale( float scale );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> setData( const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, const std::vector<cv::Point3f>& points3d, const cv::Mat& descriptors );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> setData( const cv::Mat& img, const Features& features );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       optimizePose();
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       getPoseSVD( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current );
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d>       getPoseG2O( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current );

  bool    getInitializedFlag() const;
  cv::Mat getDebugImage() const;
};


}  // namespace super_vio