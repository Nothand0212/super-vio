#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "logger/logger.h"
#include "super_vio/base_onnx_runner.h"
#include "super_vio/extractor.h"
#include "super_vio/frame.h"
#include "super_vio/matcher.h"
#include "utilities/accumulate_average.h"
#include "utilities/configuration.h"
#include "utilities/image_process.h"
#include "utilities/read_kitii_dataset.hpp"
#include "utilities/reconstructor.h"
#include "utilities/timer.h"
#include "utilities/visualizer.h"

void publishPointCloud( ros::Publisher& pub, const std::vector<Eigen::Vector3d>& points )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  // Fill the PointCloud with the points
  for ( const auto& point : points )
  {
    cloud->points.push_back( pcl::PointXYZ( point[ 0 ], point[ 2 ], -point[ 1 ] ) );
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

  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>( "/super_vio/point_cloud", 1 );

  image_transport::ImageTransport it( nh );
  image_transport::Publisher      image_pub = it.advertise( "/super_vio/image", 1 );


  std::string config_path;
  if ( argc != 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " <path_to_config>\n";
    std::cout << BOLDRED << "Using default config path: /home/lin/Projects/super-vio/config/param.json" << RESET << "\n";
    config_path = "/home/lin/Projects/catkin_ws/src/super-vio/config/param.json";
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

  std::shared_ptr<super_vio::Extracotr> extractor_left_ptr = std::make_unique<super_vio::Extracotr>( 6, 200 );
  extractor_left_ptr->initOrtEnv( cfg );
  std::shared_ptr<super_vio::Extracotr> extractor_right_ptr = std::make_unique<super_vio::Extracotr>( 6, 200 );
  extractor_right_ptr->initOrtEnv( cfg );

  // matcher init
  std::unique_ptr<super_vio::Matcher> matcher_ptr = std::make_unique<super_vio::Matcher>();
  matcher_ptr->initOrtEnv( cfg );

  // inference
  int                          count = 0;
  double                       time_consumed;
  utilities::Timer             timer;
  utilities::Timer             test_timer;
  utilities::AccumulateAverage accumulate_average_timer;

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

    auto key_points_result_left  = left_future.get();
    auto key_points_result_right = right_future.get();
    INFO( super_vio::logger, "Both Inference time: {0}", test_timer.tocGetDuration() );
    // async end


    auto key_points_src = key_points_result_left.getKeyPoints();
    auto key_points_dst = key_points_result_right.getKeyPoints();

    float scale_temp = extractor_left_ptr->getScale();
    matcher_ptr->setParams( std::vector<float>( scale_temp, scale_temp ), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f );
    auto matches_set = matcher_ptr->inferenceDescriptorPair( cfg, key_points_src, key_points_dst, key_points_result_left.getDescriptor(), key_points_result_right.getDescriptor() );


    std::vector<cv::Point2f> key_points_transformed_src = getKeyPointsInOriginalImage( key_points_src, scale_temp );
    std::vector<cv::Point2f> key_points_transformed_dst = getKeyPointsInOriginalImage( key_points_dst, scale_temp );

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


    // Triangulate keypoints
    // pixel to camera
    test_timer.tic();
    cv::Mat K_left  = cfg.camera_matrix_left;
    cv::Mat K_right = cfg.camera_matrix_right;

    Eigen::Matrix<double, 3, 4> pose_left  = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose_right = Eigen::Matrix<double, 3, 4>::Identity();
    pose_right( 0, 3 )                     = -386.1448 / 718.856;  //-0.12f;  // 120mm

    std::vector<Eigen::Vector3d> points_3d;
    for ( const auto& match : matches_set )
    {
      Eigen::Vector3d point_3d;

      bool success = utilities::compute3DPoint( K_left, K_right, pose_left, pose_right, key_points_transformed_src[ match.first ], key_points_transformed_dst[ match.second ], point_3d );

      // bool success = utilities::triangulate( pose_left, pose_right, key_points_transformed_src[ match.first ], key_points_transformed_dst[ match.second ], point_3d );

      if ( !success )
      {
        WARN( super_vio::logger, "Triangulate failed" );
        continue;
      }
      else if ( point_3d[ 2 ] < 0 )
      {
        WARN( super_vio::logger, "Triangulate failed, point behind camera" );
        continue;
      }
      else
      {
        // INFO( super_vio::logger, "Triangulate success. Point: [{0}, {1}, {2}]", point_3d[ 0 ], point_3d[ 1 ], point_3d[ 2 ] );
        points_3d.push_back( point_3d );
      }
    }
    INFO( super_vio::logger, "Triangulate time: {0}", test_timer.tocGetDuration() );

    // std::thread visualizer_3d( [ points_3d ]() {
    //   visualizePoints( points_3d );
    // } );
    // visualizer_3d.detach();

    test_timer.tic();
    publishPointCloud( cloud_pub, points_3d );


    auto img = visualizeMatches( img_left, img_right, matches_pair, key_points_transformed_src, key_points_transformed_dst );
    publishImage( image_pub, img );
    test_timer.toc();
    INFO( super_vio::logger, "visualize time: {0}", test_timer.tocGetDuration() );
  }
  return 0;
}
