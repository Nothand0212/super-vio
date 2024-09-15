#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "logger/logger.h"
#include "super_vio/base_onnx_runner.h"
#include "super_vio/camera.hpp"
#include "super_vio/extractor.h"
#include "super_vio/frame.h"
#include "super_vio/map_point.hpp"
#include "super_vio/matcher.h"
#include "super_vio/pose_estimator_3d3d.h"
#include "utilities/accumulate_average.h"
#include "utilities/color.h"
#include "utilities/configuration.h"
#include "utilities/image_process.h"
#include "utilities/read_kitii_dataset.hpp"
#include "utilities/reconstructor.h"
#include "utilities/timer.h"
#include "utilities/visualizer.h"

void publishPointCloud( ros::Publisher& pub, const std::vector<Eigen::Vector3d>& points, const Eigen::Matrix3d& cumulative_rotation, const Eigen::Vector3d& cumulative_translation )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  // Transform each point to the first frame coordinate system
  for ( const auto& point : points )
  {
    // Apply the cumulative rotation and translation
    Eigen::Vector3d transformed_point = cumulative_rotation * point + cumulative_translation;
    cloud->points.push_back( pcl::PointXYZ( transformed_point[ 0 ], transformed_point[ 1 ], transformed_point[ 2 ] ) );
  }

  // Convert the PointCloud to a sensor_msgs/PointCloud2
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg( *cloud, output );
  output.header.frame_id = "camera_link";
  output.header.stamp    = ros::Time::now();

  // Publish the data
  pub.publish( output );
}

void publishPointCloud( ros::Publisher& pub, const std::vector<Eigen::Vector3d>& points )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  // Transform each point to the first frame coordinate system
  for ( const auto& point : points )
  {
    cloud->points.push_back( pcl::PointXYZ( point[ 0 ], point[ 1 ], point[ 2 ] ) );
  }

  // Convert the PointCloud to a sensor_msgs/PointCloud2
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg( *cloud, output );
  output.header.frame_id = "camera_link";
  output.header.stamp    = ros::Time::now();

  // Publish the data
  pub.publish( output );
}

void publishImage( image_transport::Publisher& pub, const cv::Mat& image )
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage( std_msgs::Header(), "bgr8", image ).toImageMsg();
  // msg.header.stamp          = ros::Time::now();

  // msg.header.frame_id = "camera_link";
  pub.publish( msg );
}

int main( int argc, char** argv )
{
  ros::init( argc, argv, "offline_node", ros::init_options::NoSigintHandler );
  ros::NodeHandle nh;

  ros::Publisher cloud_pub   = nh.advertise<sensor_msgs::PointCloud2>( "/super_vio/point_cloud", 1 );
  ros::Publisher cloud_pub_2 = nh.advertise<sensor_msgs::PointCloud2>( "/super_vio/point_cloud_2", 1 );

  image_transport::ImageTransport it( nh );
  image_transport::Publisher      image_pub = it.advertise( "/super_vio/image", 1 );


  std::string config_path;
  if ( argc != 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " <path_to_config>\n";
    std::cout << BOLDRED << "Using default config path: /home/lin/Projects/ss_ws/src/super-vio/config/param.json" << RESET << "\n";
    config_path = "/home/lin/Projects/ss_ws/src/super-vio/config/param.json";
  }
  else
  {
    config_path = argv[ 1 ];
  }

  utilities::Configuration cfg{};
  cfg.readConfigFile( config_path );

  auto new_cfg = cfg;

  super_vio::initLogger( cfg.log_path );
  INFO( super_vio::logger, "Start" );


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
  int                          count = 0;
  double                       time_consumed;
  utilities::Timer             timer;
  utilities::Timer             test_timer;
  utilities::AccumulateAverage accumulate_average_timer;

  // 初始化第一帧的累积变换
  Eigen::Matrix3d cumulative_rotation    = Eigen::Matrix3d::Identity();
  Eigen::Vector3d cumulative_translation = Eigen::Vector3d::Zero();

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
    auto left_future = std::async( std::launch::async, [ extractor_left_ptr, &cfg, &img_left ]() {
      return extractor_left_ptr->inferenceImage( cfg, img_left );
    } );

    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, &new_cfg, &img_right ]() {
      return extractor_right_ptr->inferenceImage( new_cfg, img_right );
    } );

    auto features_on_left_img  = left_future.get();
    auto features_on_right_img = right_future.get();
    INFO( super_vio::logger, "Both Inference time: {0}", test_timer.tocGetDuration() );
    // async end


    auto key_points_left   = features_on_left_img.getKeyPoints();
    auto key_points_right  = features_on_right_img.getKeyPoints();
    auto descriptors_left  = features_on_left_img.getDescriptor();
    auto descriptors_right = features_on_right_img.getDescriptor();

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
    INFO( super_vio::logger, "Pipline Time Consumed: {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );
    INFO( super_vio::logger, "Key Points Number: {0} / {1}", key_points_transformed_src.size(), key_points_transformed_dst.size() );


    std::vector<std::pair<int, int>> triangular_matches;
    std::vector<bool>                triangular_success( matches_set.size(), false );
    std::vector<Eigen::Vector3d>     points_3d{ matches_set.size() };

    // for test
    std::vector<cv::Point2f> pixel_left, pixel_right;

    // end for test

    test_timer.tic();
    std::size_t match_idx{ 0 };
    for ( const auto& match : matches_set )
    {
      Eigen::Vector3d point_3d = Eigen::Vector3d::Zero();


      bool success = utilities::compute3DPoint( camera_params_left, camera_params_right, pose_left, pose_right, key_points_transformed_src[ match.first ], key_points_transformed_dst[ match.second ], point_3d );


      if ( !success )
      {
        triangular_success[ match_idx ] = false;
        WARN( super_vio::logger, "Triangulate failed" );
      }
      else
      {
        triangular_success[ match_idx ] = true;
        pixel_left.push_back( key_points_transformed_src[ match.first ] );
        pixel_right.push_back( key_points_transformed_dst[ match.second ] );
        // triangular_matches.push_back( match );
        INFO( super_vio::logger, "Triangulate success. {2} KeyPoint: [{0}, {1}]", key_points_transformed_src[ match.first ].x, key_points_transformed_src[ match.first ].y, match_idx );
        INFO( super_vio::logger, "Point: [{0}, {1}, {2}]", point_3d[ 0 ], point_3d[ 1 ], point_3d[ 2 ] );
      }

      points_3d[ match_idx ] = point_3d;
      triangular_matches.push_back( match );
      match_idx++;
    }

    // std::vector<Eigen::Vector3d> point_3d_test;
    // utilities::compute3DPoints( camera_params_left, camera_params_right, pose_left, pose_right, pixel_left, pixel_right, point_3d_test );
    // for ( std::size_t i = 0; i < point_3d_test.size(); i++ )
    // {
    //   INFO( super_vio::logger, "Point3D-Test: {0} {1} {2}", point_3d_test[ i ][ 0 ], point_3d_test[ i ][ 1 ], point_3d_test[ i ][ 2 ] );
    //   INFO( super_vio::logger, "Point3D:      {0} {1} {2}", points_3d[ i ][ 0 ], points_3d[ i ][ 1 ], points_3d[ i ][ 2 ] );
    // }

    // TODO:
    // 1. 特征和3D点的对应关系
    // 2. Feature和Features分不清了
    // for ( std::size_t i = 0; i < triangular_matches.size(); i++ )
    // {
    //   // 假设每一对特征点都能成功匹配到3D点
    //   std::shared_ptr<super_vio::MapPoint> map_point_ptr( new super_vio::MapPoint );
    //   map_point_ptr->setPosition( point_3d_test[ i ] );
    //   features_on_left_img[ triangular_matches[ i ].first ].setMapPoint( map_point_ptr );
    //   features_on_right_img[ triangular_matches[ i ].second ].setMapPoint( map_point_ptr );
    // }

    INFO( super_vio::logger, "Triangulate {0} Points consumed time: {1}", triangular_matches.size(), test_timer.tocGetDuration() );
    if ( triangular_matches.size() != matches_set.size() )
    {
      ERROR( super_vio::logger, "triangular_matches.size(){0} !=  matches_set.size(){1}", triangular_matches.size(), matches_set.size() );
      return -1;
    }

    // extract keypoints and descriptors from triangular matches
    std::vector<cv::Point2f> key_points_left_triangular;
    std::vector<cv::Point3f> points_3d_cv_triangular;
    cv::Mat                  descriptors_left_triangular;
    descriptors_left_triangular.reserve( triangular_matches.size() );
    std::ostringstream keypoint_3dpoint_oss;
    for ( std::size_t i = 0; i < triangular_matches.size(); i++ )
    {
      if ( triangular_success[ i ] == true )
      {
        key_points_left_triangular.emplace_back( key_points_left[ triangular_matches[ i ].first ] );
        // points_3d_cv_triangular.emplace_back( cv::Point3f( points_3d[ triangular_matches[ i ].first ][ 0 ],
        //                                                    points_3d[ triangular_matches[ i ].first ][ 1 ],
        //                                                    points_3d[ triangular_matches[ i ].first ][ 2 ] ) );
        points_3d_cv_triangular.emplace_back( cv::Point3f( points_3d[ i ][ 0 ],
                                                           points_3d[ i ][ 1 ],
                                                           points_3d[ i ][ 2 ] ) );
        descriptors_left_triangular.push_back( descriptors_left.row( triangular_matches[ i ].first ) );
        keypoint_3dpoint_oss << "\n\nIndex: " << i;
        keypoint_3dpoint_oss << "\nKeyPoint: " << key_points_transformed_src[ triangular_matches[ i ].first ].x << " " << key_points_transformed_src[ triangular_matches[ i ].first ].y;
        keypoint_3dpoint_oss << "\nPoint3D: " << points_3d_cv_triangular.back().[ 0 ] << " " << points_3d_cv_triangular.back().[ 1 ] << " " << points_3d_cv_triangular.back().[ 2 ];
      }
    }
    INFO( super_vio::logger, keypoint_3dpoint_oss.str() );
    INFO( super_vio::logger, "Descriptors Number: {0} / {1}", descriptors_left_triangular.rows, descriptors_left.rows );

    // for ( std::size_t i = 0; i < features_on_left_img.size(); i++ )
    // {
    //   if ( features_on_left_img[ i ].getMapPoint() != nullptr )
    //   {
    //     key_points_left_triangular.emplace_back( features_on_left_img[ i ].get );
    //   }
    // }


    // std::size_t test_1{ 0 };
    // for ( std::size_t i = 0; i < triangular_matches.size(); i++ )
    // {
    //   std::ostringstream oss;
    //   if ( triangular_success[ i ] == true )
    //   {
    //     oss << BOLDGREEN << "\nTriangular Descriptor: ";
    //     for ( std::size_t j = 0; j < descriptors_left_triangular.row( test_1 ).cols; j++ )
    //     {
    //       oss << descriptors_left_triangular.at<float>( test_1, j ) << " ";
    //     }
    //     oss << RESET;

    //     test_1++;

    //     oss << BOLDCYAN << "\nOriginal Descriptor: ";
    //     for ( std::size_t j = 0; j < descriptors_left.row( triangular_matches[ i ].first ).cols; j++ )
    //     {
    //       oss << descriptors_left.at<float>( triangular_matches[ i ].first, j ) << " ";
    //     }
    //     oss << RESET;
    //     INFO( super_vio::logger, oss.str() );
    //   }
    // }

    auto [ rotation, translation, success ] = pose_estimator_ptr->setData( img_left, key_points_left_triangular, points_3d_cv_triangular, descriptors_left_triangular );
    if ( ni != 0 )
    {
      cumulative_rotation    = cumulative_rotation * rotation;
      cumulative_translation = cumulative_rotation * translation + cumulative_translation;
    }

    test_timer.tic();
    publishPointCloud( cloud_pub, points_3d, cumulative_rotation, cumulative_translation );
    publishPointCloud( cloud_pub_2, points_3d );


    auto img = visualizeMatches( img_left, img_right, matches_pair, key_points_transformed_src, key_points_transformed_dst );
    publishImage( image_pub, img );
    test_timer.toc();
    INFO( super_vio::logger, "visualize time: {0}", test_timer.tocGetDuration() );
  }
  return 0;
}
