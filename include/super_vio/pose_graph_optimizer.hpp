#pragma once
/**
 * @file optimizer.hpp
 * @author LinZeshi (linzeshi@foxmail.com)
 * @brief todo.
 * 1. After detect Loop, should optimize the whole pose graph
 * @version 0.1
 * @date 2024-10-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>

#include "logger/mine_logger.hpp"

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv( const Sophus::SE3d &e )
{
  Matrix6d J;
  J.block( 0, 0, 3, 3 ) = Sophus::SO3d::hat( e.so3().log() );
  J.block( 0, 3, 3, 3 ) = Sophus::SO3d::hat( e.translation() );
  J.block( 3, 0, 3, 3 ) = Eigen::Matrix3d::Zero( 3, 3 );
  J.block( 3, 3, 3, 3 ) = Sophus::SO3d::hat( e.so3().log() );
  J                     = J * 0.5 + Matrix6d::Identity();
  //   J = Matrix6d::Identity();  // try Identity if you want
  return J;
}

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual bool read( std::istream &is ) override
  {
    double data[ 7 ];
    for ( int i = 0; i < 7; i++ )
      is >> data[ i ];
    setEstimate( Sophus::SE3d(
        Eigen::Quaterniond( data[ 6 ], data[ 3 ], data[ 4 ], data[ 5 ] ),
        Eigen::Vector3d( data[ 0 ], data[ 1 ], data[ 2 ] ) ) );
  }

  virtual bool write( std::ostream &os ) const override
  {
    os << id() << " ";
    Eigen::Quaterniond q = _estimate.unit_quaternion();
    os << _estimate.translation().transpose() << " ";
    os << q.coeffs()[ 0 ] << " " << q.coeffs()[ 1 ] << " " << q.coeffs()[ 2 ] << " " << q.coeffs()[ 3 ] << std::endl;
    return true;
  }

  virtual void setToOriginImpl() override
  {
    _estimate = Sophus::SE3d();
  }

  // 左乘更新
  virtual void oplusImpl( const double *update ) override
  {
    Vector6d upd;
    upd << update[ 0 ], update[ 1 ], update[ 2 ], update[ 3 ], update[ 4 ], update[ 5 ];
    _estimate = Sophus::SE3d::exp( upd ) * _estimate;
  }
};

// 两个李代数节点之边
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual bool read( std::istream &is ) override
  {
    double data[ 7 ];
    for ( int i = 0; i < 7; i++ )
      is >> data[ i ];
    Eigen::Quaterniond q( data[ 6 ], data[ 3 ], data[ 4 ], data[ 5 ] );
    q.normalize();
    setMeasurement( Sophus::SE3d( q, Eigen::Vector3d( data[ 0 ], data[ 1 ], data[ 2 ] ) ) );
    for ( int i = 0; i < information().rows() && is.good(); i++ )
      for ( int j = i; j < information().cols() && is.good(); j++ )
      {
        is >> information()( i, j );
        if ( i != j )
          information()( j, i ) = information()( i, j );
      }
    return true;
  }

  virtual bool write( std::ostream &os ) const override
  {
    VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *>( _vertices[ 0 ] );
    VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *>( _vertices[ 1 ] );
    os << v1->id() << " " << v2->id() << " ";
    Sophus::SE3d       m = _measurement;
    Eigen::Quaterniond q = m.unit_quaternion();
    os << m.translation().transpose() << " ";
    os << q.coeffs()[ 0 ] << " " << q.coeffs()[ 1 ] << " " << q.coeffs()[ 2 ] << " " << q.coeffs()[ 3 ] << " ";

    // information matrix
    for ( int i = 0; i < information().rows(); i++ )
      for ( int j = i; j < information().cols(); j++ )
      {
        os << information()( i, j ) << " ";
      }
    os << std::endl;
    return true;
  }

  // 误差计算与书中推导一致
  virtual void computeError() override
  {
    Sophus::SE3d v1 = ( static_cast<VertexSE3LieAlgebra *>( _vertices[ 0 ] ) )->estimate();
    Sophus::SE3d v2 = ( static_cast<VertexSE3LieAlgebra *>( _vertices[ 1 ] ) )->estimate();
    _error          = ( _measurement.inverse() * v1.inverse() * v2 ).log();
  }

  // 雅可比计算
  virtual void linearizeOplus() override
  {
    Sophus::SE3d v1 = ( static_cast<VertexSE3LieAlgebra *>( _vertices[ 0 ] ) )->estimate();
    Sophus::SE3d v2 = ( static_cast<VertexSE3LieAlgebra *>( _vertices[ 1 ] ) )->estimate();
    Matrix6d     J  = JRInv( Sophus::SE3d::exp( _error ) );
    // 尝试把J近似为I？
    _jacobianOplusXi = -J * v2.inverse().Adj();
    _jacobianOplusXj = J * v2.inverse().Adj();
  }
};

namespace super_vio
{
class PoseGraphOptimizer
{
  using BlockSolverType  = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
  using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

private:
  g2o::SparseOptimizer               optimizer_;
  std::vector<VertexSE3LieAlgebra *> vectices_;
  int                                vertex_num_{ 0 };
  int                                edge_num_{ 0 };

public:
  PoseGraphOptimizer();
  void                      addVertex( const Sophus::SE3d &pose );
  void                      addEdge( int id1, int id2 );
  void                      optimize( int max_iterations = 30 );
  std::vector<Sophus::SE3d> getOptimizedPoses() const;
};


}  // namespace super_vio