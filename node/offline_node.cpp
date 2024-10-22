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

#include "logger/mine_logger.hpp"
#include "super_vio/base_onnx_runner.hpp"
#include "super_vio/extractor.hpp"
#include "super_vio/frame.hpp"
#include "super_vio/matcher.hpp"
#include "utilities/accumulate_average.hpp"
#include "utilities/configuration.hpp"
#include "utilities/image_process.hpp"
#include "utilities/reconstructor.hpp"
#include "utilities/timer.hpp"
#include "utilities/visualizer.hpp"

std::vector<cv::Mat> readImage( std::vector<cv::String> image_file_vec, bool grayscale = false )
{
  /*
    Func:
        Read an image from path as RGB or grayscale

    */
  int mode = cv::IMREAD_COLOR;
  if ( grayscale )
  {
    mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
  }

  std::vector<cv::Mat> image_matlist;
  for ( const auto& file : image_file_vec )
  {
    cv::Mat image = cv::imread( file, mode );
    if ( image.empty() )
    {
      throw std::runtime_error( "[ERROR] Could not read image at " + file );
    }
    if ( !grayscale )
    {
      cv::cvtColor( image, image, cv::COLOR_BGR2RGB );  // BGR -> RGB
    }
    image_matlist.emplace_back( image );
  }

  return image_matlist;
}

void publishPointCloud( ros::Publisher& pub, const std::vector<Eigen::Vector3d>& points )
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  // Fill the PointCloud with the points
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

  super_vio::initLogger( cfg.log_path );
  INFO( super_vio::logger, "Start" );

  utilities::Timer             timer;
  utilities::AccumulateAverage accumulate_average_timer;

  std::vector<cv::String> image_file_src_vec;
  std::vector<cv::String> image_file_dst_vec;

  // Read image file path
  cv::glob( cfg.image_src_path, image_file_src_vec );
  cv::glob( cfg.image_dst_path, image_file_dst_vec );

  // Read image
  if ( image_file_src_vec.size() != image_file_dst_vec.size() )
  {
    ERROR( super_vio::logger, "image src number: {0}", image_file_src_vec.size() );
    ERROR( super_vio::logger, "image dst number: {0}", image_file_dst_vec.size() );
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_src_mat_vec = readImage( image_file_src_vec, cfg.gray_flag );
  std::vector<cv::Mat> image_dst_mat_vec = readImage( image_file_dst_vec, cfg.gray_flag );

  std::shared_ptr<super_vio::Extractor> extractor_left_ptr = std::make_unique<super_vio::Extractor>( 6, 200 );
  extractor_left_ptr->initOrtEnv( cfg );
  std::shared_ptr<super_vio::Extractor> extractor_right_ptr = std::make_unique<super_vio::Extractor>( 6, 200 );
  extractor_right_ptr->initOrtEnv( cfg );

  // matcher init
  std::unique_ptr<super_vio::Matcher> matcher_ptr = std::make_unique<super_vio::Matcher>();
  matcher_ptr->initOrtEnv( cfg );

  // inference
  int    count = 0;
  double time_consumed;
  auto   iter_src = image_src_mat_vec.begin();
  auto   iter_dst = image_dst_mat_vec.begin();
  for ( ; iter_src != image_src_mat_vec.end(); ++iter_src, ++iter_dst )
  {
    count++;
    timer.tic();

    auto left_future = std::async( std::launch::async, [ extractor_left_ptr, cfg, iter_src ]() {
      return extractor_left_ptr->inferenceImage( cfg, *iter_src );
    } );

    auto right_future = std::async( std::launch::async, [ extractor_right_ptr, cfg, iter_dst ]() {
      return extractor_right_ptr->inferenceImage( cfg, *iter_dst );
    } );

    auto key_points_result_left  = left_future.get();
    auto key_points_result_right = right_future.get();

    auto key_points_src = key_points_result_left.getKeyPoints();
    auto key_points_dst = key_points_result_right.getKeyPoints();

    float scale_temp = extractor_left_ptr->getScale();
    matcher_ptr->setParams( std::vector<float>( scale_temp, scale_temp ), extractor_left_ptr->getHeightTransformed(), extractor_left_ptr->getWidthTransformed(), 0.0f );
    auto matches_set = matcher_ptr->inferenceDescriptorPair( cfg, key_points_src, key_points_dst, key_points_result_left.getDescriptors(), key_points_result_right.getDescriptors() );


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


    // Triangulate keypoints
    // pixel to camera
    cv::Mat K_left  = cfg.camera_matrix_left;
    cv::Mat K_right = cfg.camera_matrix_right;

    Eigen::Matrix<double, 3, 4> pose_left  = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose_right = Eigen::Matrix<double, 3, 4>::Identity();
    pose_right( 0, 3 )                     = -0.12f;  // 120mm

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
        INFO( super_vio::logger, "Triangulate success. Point: [{0}, {1}, {2}]", point_3d[ 0 ], point_3d[ 1 ], point_3d[ 2 ] );
        points_3d.push_back( point_3d );
      }
    }

    // std::thread visualizer_3d( [ points_3d ]() {
    //   visualizePoints( points_3d );
    // } );
    // visualizer_3d.detach();
    publishPointCloud( cloud_pub, points_3d );


    time_consumed = timer.tocGetDuration();
    accumulate_average_timer.addValue( time_consumed );
    INFO( super_vio::logger, "time consumed: {0} / {1}", time_consumed, accumulate_average_timer.getAverage() );
    INFO( super_vio::logger, "key points number: {0} / {1}", key_points_transformed_src.size(), key_points_transformed_dst.size() );

    auto img = visualizeMatches( *iter_src, *iter_dst, matches_pair, key_points_transformed_src, key_points_transformed_dst );
    publishImage( image_pub, img );
  }
  return 0;
}
