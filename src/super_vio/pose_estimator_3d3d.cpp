#include "super_vio/pose_estimator_3d3d.hpp"

namespace super_vio
{
cv::Point3f calculateCentroid( const std::vector<cv::Point3f>& points )
{
  cv::Point3f centroid( 0, 0, 0 );

  for ( const auto& point : points )
  {
    centroid += point;
  }

  int size = points.size();
  centroid.x /= size;
  centroid.y /= size;
  centroid.z /= size;

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

cv::Point3f transFromEigen( const Eigen::Vector3d& vec )
{
  cv::Point3f point;
  point.x = vec[ 0 ];
  point.y = vec[ 1 ];
  point.z = vec[ 2 ];
  return point;
};

PoseEstimator3D3D::PoseEstimator3D3D( std::shared_ptr<Matcher> matcher_sptr, std::shared_ptr<utilities::Configuration> config_sptr, float scale )
{
  matcher_sptr_   = matcher_sptr;
  config_sptr_    = config_sptr;
  scale_          = scale;
  is_initialized_ = false;

  // g2o related
  // auto solver = new g2o::OptimizationAlgorithmDogleg( std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() ) );
  auto solver = new g2o::OptimizationAlgorithmLevenberg( std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() ) );
  optimizer_.setAlgorithm( solver );  // 设置求解器
  optimizer_.setVerbose( true );      // 打开调试输出
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
  this->timer_.tic();
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
    INFO( super_vio::logger, "Get Pose Time Consumed: {}", this->timer_.tocGetDuration() );
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

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> PoseEstimator3D3D::setData( const cv::Mat& img, const Features& features )
{
  this->timer_.tic();

  INFO( super_vio::logger, "Received new frame, features: {0}", features.getFeatures().size() );

  if ( this->is_initialized_ == true )
  {
    this->last_image_       = this->current_image_;
    this->last_keypoints_   = this->current_keypoints_;
    this->last_points3d_    = this->current_points3d_;
    this->last_descriptors_ = this->current_descriptors_;

    this->current_image_ = img;
    this->current_keypoints_.clear();
    this->current_points3d_.clear();
    this->current_descriptors_.release();

    std::size_t num = features.getFeatures().size();
    for ( std::size_t i = 0; i < num; ++i )
    {
      if ( features.getSingleMapPoint( i ) != nullptr )
      {
        this->current_keypoints_.emplace_back( features.getSingleKeyPoint( i ) );
        this->current_points3d_.emplace_back( transFromEigen( features.getSingleMapPoint( i )->getPosition() ) );
        this->current_descriptors_.push_back( features.getSingleDescriptor( i ) );
      }
      // else
      // {
      //   WARN( super_vio::logger, "Feature without map point" );
      // }
    }

    auto [ rotation, translation ] = this->optimizePose();
    INFO( super_vio::logger, "Get Pose Time Consumed: {}", this->timer_.tocGetDuration() );
    return std::make_tuple( rotation, translation, true );
  }
  else
  {
    INFO( super_vio::logger, "Receiving the first frame" );

    this->current_image_ = img;
    this->current_keypoints_.clear();
    this->current_points3d_.clear();
    this->current_descriptors_.release();

    std::size_t num = features.getFeatures().size();
    for ( std::size_t i = 0; i < num; ++i )
    {
      if ( features.getSingleMapPoint( i ) != nullptr )
      {
        this->current_keypoints_.emplace_back( features.getSingleKeyPoint( i ) );
        this->current_points3d_.emplace_back( transFromEigen( features.getSingleMapPoint( i )->getPosition() ) );
        this->current_descriptors_.push_back( features.getSingleDescriptor( i ) );
      }
      else
      {
        WARN( super_vio::logger, "Feature without map point" );
      }
    }

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
    }

    matches_2d_src.emplace_back( key_points_transformed_src[ match.first ] );
    matches_2d_dst.emplace_back( key_points_transformed_dst[ match.second ] );
  }
  // INFO( super_vio::logger, oss.str() );
  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_2d_pair = std::make_pair( matches_2d_src, matches_2d_dst );


  INFO( super_vio::logger, "Matches: {0}, key points: {1}(last) / {2}(current)", matches_2d_pair.first.size(), key_points_transformed_src.size(), key_points_transformed_dst.size() );
  // INFO( super_vio::logger, "Last image: [{0} x {1}], Current image: [{2} x {3}]", this->last_image_.cols, this->last_image_.rows, this->current_image_.cols, this->current_image_.rows );
  debug_image_ = visualizeMatches( this->last_image_, this->current_image_, matches_2d_pair, key_points_transformed_src, key_points_transformed_dst );

  // cv::imshow( "Last-Current Matches", img );
  // cv::waitKey( 0 );

  // 3. Estimate the pose using the 3D-3D correspondences
  return this->getPoseSVD( matches_3d_src, matches_3d_dst );
  // return this->getPoseG2O( matches_3d_src, matches_3d_dst );
}


std::tuple<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator3D3D::getPoseSVD( const std::vector<cv::Point3f>& points_last, const std::vector<cv::Point3f>& points_current )
{
  utilities::Timer timer;
  timer.tic();

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

  // oss << "\nW = ";
  // oss << "\n\t" << W( 0, 0 ) << " " << W( 0, 1 ) << " " << W( 0, 2 );
  // oss << "\n\t" << W( 1, 0 ) << " " << W( 1, 1 ) << " " << W( 1, 2 );
  // oss << "\n\t" << W( 2, 0 ) << " " << W( 2, 1 ) << " " << W( 2, 2 );

  // 4. Compute SVD of W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd( W, Eigen::ComputeFullU | Eigen::ComputeFullV );

  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  // oss << "\nU = ";
  // oss << "\n\t" << U( 0, 0 ) << " " << U( 0, 1 ) << " " << U( 0, 2 );
  // oss << "\n\t" << U( 1, 0 ) << " " << U( 1, 1 ) << " " << U( 1, 2 );
  // oss << "\n\t" << U( 2, 0 ) << " " << U( 2, 1 ) << " " << U( 2, 2 );
  // oss << "\nV = ";
  // oss << "\n\t" << V( 0, 0 ) << " " << V( 0, 1 ) << " " << V( 0, 2 );
  // oss << "\n\t" << V( 1, 0 ) << " " << V( 1, 1 ) << " " << V( 1, 2 );
  // oss << "\n\t" << V( 2, 0 ) << " " << V( 2, 1 ) << " " << V( 2, 2 );


  Eigen::Matrix3d rotation = U * ( V.transpose() );
  if ( rotation.determinant() < 0 )
  {
    rotation = -rotation;
  }
  // transform to euler angle(roll, pitch, yaw)
  Eigen::Vector3d euler_angles = rotation.eulerAngles( 0, 1, 2 );
  Eigen::Vector3d translation  = Eigen::Vector3d( centroid_last.x, centroid_last.y, centroid_last.z ) - rotation * Eigen::Vector3d( centroid_current.x, centroid_current.y, centroid_current.z );

  oss << "\nPose Estimation(SVD) Time Consumed: " << timer.tocGetDuration();
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
  utilities::Timer timer;
  timer.tic();


  // vertex
  VertexPose* pose = new VertexPose();  // camera pose
  pose->setId( 0 );
  pose->setEstimate( this->last_pose_ );
  optimizer_.addVertex( pose );

  cv::Point3f centroid_last    = calculateCentroid( points_last );
  cv::Point3f centroid_current = calculateCentroid( points_current );

  EdgeProjectXYZPoseOnly* edge = new EdgeProjectXYZPoseOnly( Eigen::Vector3d( centroid_current.x, centroid_current.y, centroid_current.z ) );
  edge->setVertex( 0, pose );
  edge->setMeasurement( Eigen::Vector3d( centroid_last.x, centroid_last.y, centroid_last.z ) );
  edge->setInformation( Eigen::Matrix3d::Identity() );
  optimizer_.addEdge( edge );

  // edges
  for ( size_t i = 0; i < points_last.size(); i++ )
  {
    EdgeProjectXYZPoseOnly* edge = new EdgeProjectXYZPoseOnly( Eigen::Vector3d( points_current[ i ].x, points_current[ i ].y, points_current[ i ].z ) );
    edge->setVertex( 0, pose );
    edge->setMeasurement( Eigen::Vector3d( points_last[ i ].x, points_last[ i ].y, points_last[ i ].z ) );
    edge->setInformation( Eigen::Matrix3d::Identity() );
    optimizer_.addEdge( edge );
  }

  optimizer_.initializeOptimization();
  optimizer_.optimize( 10 );

  this->last_pose_             = pose->estimate();
  Eigen::Matrix3d rotation     = pose->estimate().rotationMatrix();
  Eigen::Vector3d translation  = pose->estimate().translation();
  Eigen::Vector3d euler_angles = rotation.eulerAngles( 0, 1, 2 );

  optimizer_.clear();

  std::ostringstream oss;
  oss << "\nPose Estimation(g2o) Time Consumed: " << timer.tocGetDuration();
  oss << "\nRotation = ";
  oss << "\n\t" << rotation( 0, 0 ) << " " << rotation( 0, 1 ) << " " << rotation( 0, 2 );
  oss << "\n\t" << rotation( 1, 0 ) << " " << rotation( 1, 1 ) << " " << rotation( 1, 2 );
  oss << "\n\t" << rotation( 2, 0 ) << " " << rotation( 2, 1 ) << " " << rotation( 2, 2 );
  oss << "\nEuler angles = " << euler_angles( 0 ) << " " << euler_angles( 1 ) << " " << euler_angles( 2 );
  oss << "\nTranslation = " << translation( 0 ) << " " << translation( 1 ) << " " << translation( 2 );
  INFO( super_vio::logger, oss.str() );
  return std::make_tuple( rotation, translation );
}

cv::Mat PoseEstimator3D3D::getDebugImage() const
{
  return debug_image_;
}


}  // namespace super_vio