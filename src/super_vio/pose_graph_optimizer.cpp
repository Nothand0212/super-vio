#include "super_vio/pose_graph_optimizer.hpp"

namespace super_vio
{
PoseGraphOptimizer::PoseGraphOptimizer()
{
  auto solver = new g2o::OptimizationAlgorithmLevenberg( std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() ) );
  optimizer_.setAlgorithm( solver );
  optimizer_.setVerbose( true );
}

void PoseGraphOptimizer::addVertex( const Sophus::SE3d &pose )
{
  VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
  v->setId( vertex_num_ );
  vertex_num_++;
  v->setEstimate( pose );
  optimizer_.addVertex( v );
  vectices_.push_back( v );

  // LOG
  auto            rotation     = pose.rotationMatrix();
  auto            translation  = pose.translation();
  Eigen::Vector3d euler_angles = rotation.eulerAngles( 0, 1, 2 );

  std::ostringstream oss;
  oss << "\nPose Graphe Vertex: " << v->id();
  oss << "\nRotation = ";
  oss << "\n\t" << rotation( 0, 0 ) << " " << rotation( 0, 1 ) << " " << rotation( 0, 2 );
  oss << "\n\t" << rotation( 1, 0 ) << " " << rotation( 1, 1 ) << " " << rotation( 1, 2 );
  oss << "\n\t" << rotation( 2, 0 ) << " " << rotation( 2, 1 ) << " " << rotation( 2, 2 );
  oss << "\nEuler angles = " << euler_angles( 0 ) << " " << euler_angles( 1 ) << " " << euler_angles( 2 );
  oss << "\nTranslation = " << translation( 0 ) << " " << translation( 1 ) << " " << translation( 2 );
  INFO( super_vio::logger, oss.str() );
}

void PoseGraphOptimizer::addEdge( int id1, int id2 )
{
  EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
  e->setId( edge_num_ );
  edge_num_++;
  e->setVertex( 0, optimizer_.vertex( id1 ) );
  e->setVertex( 1, optimizer_.vertex( id2 ) );
  Sophus::SE3d m = ( static_cast<VertexSE3LieAlgebra *>( optimizer_.vertex( id1 ) ) )->estimate().inverse() * ( static_cast<VertexSE3LieAlgebra *>( optimizer_.vertex( id2 ) ) )->estimate();
  e->setMeasurement( m );
  Matrix6d &info_mat = e->information();  // 获取信息矩阵的引用

  info_mat << 10000, 0, 0, 0, 0, 0,
      0, 10000, 0, 0, 0, 0,
      0, 0, 40000, 0, 0, 0,
      0, 0, 0, 10000, 0, 0,
      0, 0, 0, 0, 10000, 0,
      0, 0, 0, 0, 0, 40000;

  optimizer_.addEdge( e );

  // LOG
  auto               rotation     = m.rotationMatrix();
  auto               translation  = m.translation();
  Eigen::Vector3d    euler_angles = rotation.eulerAngles( 0, 1, 2 );
  std::ostringstream oss;
  oss << "\nPose Graphe Edge: " << e->id();
  oss << "\nVertex 1 = " << id1;
  oss << "\nVertex 2 = " << id2;
  oss << "\nRotation = ";
  oss << "\n\t" << rotation( 0, 0 ) << " " << rotation( 0, 1 ) << " " << rotation( 0, 2 );
  oss << "\n\t" << rotation( 1, 0 ) << " " << rotation( 1, 1 ) << " " << rotation( 1, 2 );
  oss << "\n\t" << rotation( 2, 0 ) << " " << rotation( 2, 1 ) << " " << rotation( 2, 2 );
  oss << "\nEuler angles = " << euler_angles( 0 ) << " " << euler_angles( 1 ) << " " << euler_angles( 2 );
  oss << "\nTranslation = " << translation( 0 ) << " " << translation( 1 ) << " " << translation( 2 );
  INFO( super_vio::logger, oss.str() );
}

void PoseGraphOptimizer::optimize( int max_iterations )
{
  INFO( super_vio::logger, "Pose Graph Optimizer: Starting Optimization" );
  INFO( super_vio::logger, "Pose Graph Optimizer: Number of vertices = {}", vectices_.size() );
  optimizer_.initializeOptimization();
  INFO( super_vio::logger, "Pose Graph Optimizer: Starting Optimization" );
  optimizer_.optimize( max_iterations );
  INFO( super_vio::logger, "Pose Graph Optimizer: Optimization Done" );
}

std::vector<Sophus::SE3d> PoseGraphOptimizer::getOptimizedPoses() const
{
  std::vector<Sophus::SE3d> optimized_poses;

  for ( std::size_t i = 0; i < vectices_.size(); i++ )
  {
    optimized_poses.push_back( vectices_[ i ]->estimate() );
  }

  return optimized_poses;
}

}  // namespace super_vio