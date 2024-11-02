// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

// ros
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

// Eigen
#include <Eigen/Dense>

// STD
#include <future>
#include <memory>
#include <thread>

#include "logger/mine_logger.hpp"
#include "super_vio/base_onnx_runner.hpp"
#include "super_vio/camera.hpp"
#include "super_vio/extractor.hpp"
#include "super_vio/frame.hpp"
#include "super_vio/map_point.hpp"
#include "super_vio/matcher.hpp"
#include "super_vio/pose_estimator_3d3d.hpp"
#include "super_vio/pose_graph_optimizer.hpp"
#include "super_vio/ros_tool.hpp"
#include "utilities/accumulate_average.hpp"
#include "utilities/color.hpp"
#include "utilities/configuration.hpp"
#include "utilities/image_process.hpp"
#include "utilities/read_kitii_dataset.hpp"
#include "utilities/reconstructor.hpp"
#include "utilities/timer.hpp"
#include "utilities/visualizer.hpp"


int main( int argc, char** argv )
{
  ros::init( argc, argv, "offline_node", ros::init_options::NoSigintHandler );
  ros::NodeHandle nh;

  std::string project_path  = ros::package::getPath( "super_vio" );
  std::string system_config = project_path + "/config/system_params.json";


  std::vector<std::vector<Eigen::Vector3d>> point_cloud_buffer;
  std::vector<Sophus::SE3d>                 pose_buffer;
  super_vio::PoseGraphOptimizer             pose_graph_optimizer;


  utilities::Configuration cfg{};
  cfg.readConfigFile( system_config );

  auto new_cfg = cfg;

  super_vio::initLogger( cfg.log_path );
  INFO( super_vio::logger, "Start" );

  // ROS Related
  std::string        ros_config = project_path + "/config/ros_params.json";
  super_vio::ROSTool ros_tool( nh, ros_config );

  /// load sequence frames
  std::vector<std::string> image_left_vec_path, image_right_vec_path;
  std::vector<double>      vec_timestamp;

  INFO( super_vio::logger, "Loading KITTI dataset from: {0}", cfg.kitti_path );

  utilities::LoadKittiImagesTimestamps( cfg.kitti_path, image_left_vec_path, image_right_vec_path, vec_timestamp );

  const size_t num_images = image_left_vec_path.size();
  INFO( super_vio::logger, "Num Images: {0}", num_images );

  if ( num_images != image_right_vec_path.size() || num_images != vec_timestamp.size() )
  {
    ERROR( super_vio::logger, "The number of right images is {0}, the number of timestamps is {1}.", image_right_vec_path.size(), vec_timestamp.size() );
    return -1;
  }

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
  std::ostringstream          camera_oss;
  camera_oss << "\n==== ==== ==== ==== POSE DEBUG ==== ==== ====";
  camera_oss << "\nPose Left:";
  camera_oss << "\n\t" << pose_left( 0, 0 ) << "\t" << pose_left( 0, 1 ) << "\t" << pose_left( 0, 2 ) << "\t" << pose_left( 0, 3 );
  camera_oss << "\n\t" << pose_left( 1, 0 ) << "\t" << pose_left( 1, 1 ) << "\t" << pose_left( 1, 2 ) << "\t" << pose_left( 1, 3 );
  camera_oss << "\n\t" << pose_left( 2, 0 ) << "\t" << pose_left( 2, 1 ) << "\t" << pose_left( 2, 2 ) << "\t" << pose_left( 2, 3 );
  camera_oss << "\nPose Right:";
  camera_oss << "\n\t" << pose_right( 0, 0 ) << "\t" << pose_right( 0, 1 ) << "\t" << pose_right( 0, 2 ) << "\t" << pose_right( 0, 3 );
  camera_oss << "\n\t" << pose_right( 1, 0 ) << "\t" << pose_right( 1, 1 ) << "\t" << pose_right( 1, 2 ) << "\t" << pose_right( 1, 3 );
  camera_oss << "\n\t" << pose_right( 2, 0 ) << "\t" << pose_right( 2, 1 ) << "\t" << pose_right( 2, 2 ) << "\t" << pose_right( 2, 3 );
  camera_oss << "\n==== ==== ==== ==== PARAMS DEBUG ==== ==== ==== ====";
  camera_oss << "\nCamera Matrix Left:";
  camera_oss << "\n\t"
             << "fx: " << camera_params_left.fx << "\t"
             << "fy: " << camera_params_left.fy << "\t"
             << "cx: " << camera_params_left.cx << "\t"
             << "cy: " << camera_params_left.cy;
  camera_oss << "\nDistortion Coefficients Left:";
  camera_oss << "\n\t"
             << "k1: " << camera_params_left.k1 << "\t"
             << "k2: " << camera_params_left.k2 << "\t"
             << "p1: " << camera_params_left.p1 << "\t"
             << "p2: " << camera_params_left.p2 << "\t"
             << "k3: " << camera_params_left.k3;
  camera_oss << "\nCamera Matrix Right:";
  camera_oss << "\n\t"
             << "fx: " << camera_params_right.fx << "\t"
             << "fy: " << camera_params_right.fy << "\t"
             << "cx: " << camera_params_right.cx << "\t"
             << "cy: " << camera_params_right.cy;
  camera_oss << "\nDistortion Coefficients Right:";
  camera_oss << "\n\t"
             << "k1: " << camera_params_right.k1 << "\t"
             << "k2: " << camera_params_right.k2 << "\t"
             << "p1: " << camera_params_right.p1 << "\t"
             << "p2: " << camera_params_right.p2 << "\t"
             << "k3: " << camera_params_right.k3;
  INFO( super_vio::logger, "{} {} {}", BOLDGREEN, camera_oss.str(), RESET );


  // initialize extractor
  std::shared_ptr<super_vio::Extractor> extractor_left_ptr = std::make_shared<super_vio::Extractor>( 6, 200 );
  extractor_left_ptr->initOrtEnv( cfg );
  std::shared_ptr<super_vio::Extractor> extractor_right_ptr = std::make_shared<super_vio::Extractor>( 6, 200 );
  extractor_right_ptr->initOrtEnv( cfg );

  float scale_temp = 0.4125705;

  // initialize matcher
  std::shared_ptr<super_vio::Matcher> matcher_ptr = std::make_shared<super_vio::Matcher>();
  matcher_ptr->initOrtEnv( cfg );

  // initialize pose estimator
  INFO( super_vio::logger, "PoseEstimator3D3D initializing" );
  std::shared_ptr<utilities::Configuration>     config_sptr        = std::make_shared<utilities::Configuration>( cfg );
  std::shared_ptr<super_vio::PoseEstimator3D3D> pose_estimator_ptr = std::make_shared<super_vio::PoseEstimator3D3D>( matcher_ptr, config_sptr, scale_temp );


  // inference
  double                       time_consumed;
  utilities::Timer             timer;
  utilities::Timer             test_timer;
  utilities::AccumulateAverage accumulate_average_timer;

  // 初始化第一帧的累积变换
  Eigen::Matrix3d cumulative_rotation    = Eigen::Matrix3d::Identity();
  Eigen::Vector3d cumulative_translation = Eigen::Vector3d::Zero();
  Sophus::SE3d    current_pose;

  for ( std::size_t ni = 0; ni < num_images; ni++ )
  {
    timer.tic();

    cv::Mat img_left  = cv::imread( image_left_vec_path[ ni ], cv::IMREAD_GRAYSCALE );
    cv::Mat img_right = cv::imread( image_right_vec_path[ ni ], cv::IMREAD_GRAYSCALE );
    double  timestamp = vec_timestamp[ ni ];
    if ( img_left.empty() )
    {
      SPDLOG_LOGGER_ERROR( super_vio::logger, "Failed to load image at: {0}", image_left_vec_path[ ni ] );
    }

    // async start
    test_timer.tic();
    auto left_future  = std::async( std::launch::async, [ extractor_left_ptr, &cfg, &img_left ]() { return extractor_left_ptr->inferenceImage( cfg, img_left ); } );
    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, &cfg, &img_right ]() { return extractor_right_ptr->inferenceImage( cfg, img_right ); } );

    auto features_on_left_img  = left_future.get();
    auto features_on_right_img = right_future.get();
    INFO( super_vio::logger, "Both Inference time: {0}", test_timer.tocGetDuration() );
    // async end


    auto key_points_left   = features_on_left_img.getKeyPoints();
    auto key_points_right  = features_on_right_img.getKeyPoints();
    auto descriptors_left  = features_on_left_img.getDescriptors();
    auto descriptors_right = features_on_right_img.getDescriptors();

    float scale_temp = extractor_left_ptr->getScale();
    INFO( super_vio::logger, "Scale: {0}", scale_temp );

    pose_estimator_ptr->setScale( scale_temp );
    matcher_ptr->setParams( std::vector<float>( scale_temp, scale_temp ), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f );
    std::set<std::pair<int, int>> matches_set = matcher_ptr->inferenceDescriptorPair( cfg, key_points_left, key_points_right, descriptors_left, descriptors_right );

    std::vector<cv::Point2f> key_points_transformed_src = getKeyPointsInOriginalImage( key_points_left, scale_temp );
    std::vector<cv::Point2f> key_points_transformed_dst = getKeyPointsInOriginalImage( key_points_right, scale_temp );

    // for show
    std::vector<cv::Point2f> matches_src;
    std::vector<cv::Point2f> matches_dst;
    for ( const auto& match : matches_set )
    {
      matches_src.emplace_back( key_points_transformed_src[ match.first ] );
      matches_dst.emplace_back( key_points_transformed_dst[ match.second ] );
    }
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matches_pair = std::make_pair( matches_src, matches_dst );


    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( super_vio::logger, "Pipline Time Consumed:        {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );
    INFO( super_vio::logger, "Key Points Number Left-Right: {0} / {1}", key_points_transformed_src.size(), key_points_transformed_dst.size() );


    std::vector<std::pair<int, int>> triangular_matches;
    std::vector<bool>                triangular_success( matches_set.size(), false );
    std::vector<Eigen::Vector3d>     points_3d;
    points_3d.reserve( matches_set.size() );

    // for test
    std::vector<cv::Point2f> pixel_left, pixel_right;
    // end for test

    test_timer.tic();
    std::size_t success_count = 0;
    for ( const auto& match : matches_set )
    {
      Eigen::Vector3d point_3d = Eigen::Vector3d::Zero();

      bool success = utilities::compute3DPoint( camera_params_left, camera_params_right, pose_left, pose_right, key_points_transformed_src[ match.first ], key_points_transformed_dst[ match.second ], point_3d );

      if ( !success )
      {
        // WARN( super_vio::logger, "Triangulate failed" );
        continue;
      }
      else
      {
        // Test Map Point
        std::shared_ptr<super_vio::MapPoint> map_point_ptr( new super_vio::MapPoint );
        map_point_ptr->setPosition( point_3d );

        features_on_left_img.setSingleMapPoint( match.first, map_point_ptr );
        features_on_right_img.setSingleMapPoint( match.second, map_point_ptr );

        points_3d.push_back( map_point_ptr->getPosition() );
        success_count++;
      }
    }
    INFO( super_vio::logger, "Triangulate {} / {} --> Time Consumed: {}", success_count, matches_set.size(), test_timer.tocGetDuration() );


    auto [ pose_increment, success ] = pose_estimator_ptr->setData( img_left, features_on_left_img );
    current_pose                     = current_pose * pose_increment;

    test_timer.tic();
    auto img = visualizeMatches( img_left, img_right, matches_pair, key_points_transformed_src, key_points_transformed_dst );
    ros_tool.publishCurrentPointCloud( points_3d, current_pose.rotationMatrix(), current_pose.translation() );
    ros_tool.publishPose( current_pose );
    ros_tool.publishImage( img );


    pose_buffer.push_back( current_pose );
    point_cloud_buffer.push_back( points_3d );


    test_timer.toc();
    INFO( super_vio::logger, "visualize time: {0}", test_timer.tocGetDuration() );
  }

  // Below should move to Loop Detection and add reconized-pair frames
  utilities::Timer timer_pose_graph_optimizer;
  timer_pose_graph_optimizer.tic();
  INFO( super_vio::logger, "Pose Graph Optimizer initializing with Pose size: {}", pose_buffer.size() );
  for ( std::size_t i = 0; i < pose_buffer.size(); i++ )
  {
    pose_graph_optimizer.addVertex( pose_buffer[ i ] );
    if ( i >= 1 )
    {
      pose_graph_optimizer.addEdge( i - 1, i );
    }
  }

  INFO( super_vio::logger, "Pose Graph Optimizer running" );
  pose_graph_optimizer.optimize();
  std::vector<Sophus::SE3d> poses = pose_graph_optimizer.getOptimizedPoses();
  timer_pose_graph_optimizer.toc();
  INFO( super_vio::logger, "Pose Graph Optimizer Time Consumed: {0}", timer_pose_graph_optimizer.tocGetDuration() );

  ros::Publisher cloud_pub_3 = nh.advertise<sensor_msgs::PointCloud2>( "/super_vio/global_map", 1 );

  ros_tool.publishGlobalPointCloud( point_cloud_buffer, poses );
  INFO( super_vio::logger, "Global Map Published" );

  while ( true )
  {
    ros_tool.publishGlobalPointCloud( point_cloud_buffer, poses );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
  }

  return 0;
}
