#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>

#include "common.h"
#include "logger/mine_logger.hpp"
#include "sophus/se3.hpp"


// define vertex for pose
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override
  {
    _estimate = Sophus::SE3d();
  }

  virtual void oplusImpl( const double *update ) override
  {
    Eigen::Matrix<double, 6, 1> update_mat;
    update_mat << update[ 0 ], update[ 1 ], update[ 2 ], update[ 3 ], update[ 4 ], update[ 5 ];
    _estimate = Sophus::SE3d::exp( update_mat ) * _estimate;
  }

  virtual bool read( std::istream &in ) override
  {
    return true;
  }

  virtual bool write( std::ostream &out ) const override
  {
    return true;
  }
};

// map point（3D xyz） vertex
class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexXYZ() {}

  virtual void setToOriginImpl() override
  {
    _estimate = Eigen::Vector3d::Zero();
  }

  virtual void oplusImpl( const double *update ) override
  {
    _estimate[ 0 ] += update[ 0 ];
    _estimate[ 1 ] += update[ 1 ];
    _estimate[ 2 ] += update[ 2 ];
  }

  virtual bool read( std::istream &in ) override
  {
    return true;
  }

  virtual bool write( std::ostream &out ) const override
  {
    return true;
  }
};

class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection( const Eigen::Matrix3d &K, const Sophus::SE3d &cam_pose ) : _K( K ), _pose( cam_pose ) {}

  virtual void computeError() override
  {
    const VertexPose *v0        = static_cast<VertexPose *>( _vertices[ 0 ] );
    const VertexXYZ * v1        = static_cast<VertexXYZ *>( _vertices[ 1 ] );
    Sophus::SE3d      T         = v0->estimate();
    Eigen::Vector3d   pos_pixel = _K * ( _pose * ( T * v1->estimate() ) );
    pos_pixel /= pos_pixel[ 2 ];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual bool read( std::istream &in ) override { return true; }

  virtual bool write( std::ostream &out ) const override { return true; }

private:
  Eigen::Matrix3d _K;
  Sophus::SE3d    _pose;
};

constexpr double chi2_th = 5.891;

void constructEdgeProjection( EdgeProjection *edge, VertexPose *pose, VertexXYZ *point, const Eigen::Vector2d pixel, const std::size_t index )
{
  edge->setId( index );
  edge->setVertex( 0, pose );
  edge->setVertex( 1, point );
  edge->setMeasurement( pixel );
  edge->setInformation( Eigen::Matrix2d::Identity() );
  auto rk = new g2o::RobustKernelHuber();
  rk->setDelta( chi2_th );
  edge->setRobustKernel( rk );
}

namespace super_vio
{
class BundleAdjustmentor
{
  using BlockSolverType  = g2o::BlockSolver_6_3;
  using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

private:
  g2o::SparseOptimizer optimizer_;

  std::size_t edge_idx_{ 0 };

public:
  BundleAdjustmentor();
  void addEdge( EdgeProjection *edge );
};

BundleAdjustmentor::BundleAdjustmentor()
{
  auto solver = new g2o::OptimizationAlgorithmLevenberg( std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() ) );
  optimizer_.setAlgorithm( solver );
}

void BundleAdjustmentor::addEdge( EdgeProjection *edge )
{
  optimizer_.addEdge( edge );
}


}  // namespace super_vio