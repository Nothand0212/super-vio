/**
 ******************************************************************************
 * @file           : include/visualizer.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#pragma once
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "data/features.h"

void visualizeMatches( const cv::Mat& src, const cv::Mat& dst, const std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& key_points, const std::vector<cv::Point2f>& key_points_src, const std::vector<cv::Point2f>& key_points_dst )
{
  // Convert the points to cv::KeyPoint objects
  std::vector<cv::KeyPoint> key_points_1, key_points_2;
  for ( const auto& point : key_points.first )
  {
    key_points_1.push_back( cv::KeyPoint( point, 1.0f ) );
  }
  for ( const auto& point : key_points.second )
  {
    key_points_2.push_back( cv::KeyPoint( point, 1.0f ) );
  }

  // Create cv::DMatch objects for each pair of points
  std::vector<cv::DMatch> matches;
  for ( size_t i = 0; i < key_points_1.size(); ++i )
  {
    matches.push_back( cv::DMatch( i, i, 0 ) );
  }

  // Draw the matches
  cv::Mat img_matches;
  cv::drawMatches( src, key_points_1, dst, key_points_2, matches, img_matches );

  for ( const auto& point : key_points_src )
  {
    cv::circle( img_matches, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& point : key_points_dst )
  {
    cv::circle( img_matches, cv::Point2f( point.x + src.cols, point.y ), 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  // Display the matches
  cv::imshow( "Matches", img_matches );
  cv::waitKey( 0 );
}

void visualizeKeyPoints( const cv::Mat& image_src, const cv::Mat& image_dst, const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst )
{
  cv::Mat img_src_color, img_dst_color;
  cv::cvtColor( image_src, img_src_color, cv::COLOR_GRAY2BGR );
  cv::cvtColor( image_dst, img_dst_color, cv::COLOR_GRAY2BGR );

  cv::Mat img_matches;
  cv::hconcat( img_src_color, img_dst_color, img_matches );

  for ( const auto& point : key_points_src )
  {
    cv::circle( img_matches, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& point : key_points_dst )
  {
    cv::circle( img_matches, cv::Point2f( point.x + image_src.cols, point.y ), 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  // Display the matches
  cv::imshow( "Keypoints", img_matches );
  cv::waitKey( 0 );
}

void visualizeKeyPoints( const cv::Mat& image_src, const cv::Mat& image_dst,
                         const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst,
                         std::vector<Region> regions_src, std::vector<Region> regions_dst )
{
  cv::Mat img_src_color, img_dst_color;
  cv::cvtColor( image_src, img_src_color, cv::COLOR_GRAY2BGR );
  cv::cvtColor( image_dst, img_dst_color, cv::COLOR_GRAY2BGR );


  for ( const auto& point : key_points_src )
  {
    cv::circle( img_src_color, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& point : key_points_dst )
  {
    cv::circle( img_dst_color, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& region : regions_src )
  {
    cv::rectangle( img_src_color, region.rectangle, cv::Scalar( 0, 255, 0 ), 1 );
  }

  for ( const auto& region : regions_dst )
  {
    cv::rectangle( img_dst_color, region.rectangle, cv::Scalar( 0, 255, 0 ), 1 );
  }

  cv::Mat img_matches;
  cv::hconcat( img_src_color, img_dst_color, img_matches );

  // Display the matches
  cv::imshow( "Keypoints", img_matches );
  cv::waitKey( 0 );
}


inline void visualizePoints( const std::vector<Eigen::Vector3d>& points )
{
  // Create a PointCloud object
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

  // Fill the PointCloud with the points
  for ( const auto& point : points )
  {
    cloud->points.push_back( pcl::PointXYZ( point[ 0 ], point[ 1 ], point[ 2 ] ) );
  }

  // Create a viewer
  pcl::visualization::PCLVisualizer viewer( "Cloud Viewer" );

  // Set the background color
  viewer.setBackgroundColor( 0, 0, 0 );

  // Add the point cloud to the viewer
  viewer.addPointCloud<pcl::PointXYZ>( cloud, "sample cloud" );

  // Set the point size
  viewer.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud" );

  // Spin the viewer until the user closes the window

  while ( !viewer.wasStopped() )
  {
    viewer.spinOnce( 100 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
  }
}