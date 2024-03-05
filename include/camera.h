#pragma once

#include <fstream>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

class CameraDriver
{
public:
  CameraDriver();
  ~CameraDriver();

  void loadConfig( const std::string& config_file_path );
  void open( const int& device );
  void initialize();
  void close();
  void getFrame( cv::Mat& frame );
  void getStereoFrame( cv::Mat& frame_left, cv::Mat& frame_right );
  void getStereoFrameUndistorted( cv::Mat& frame_left, cv::Mat& frame_right );

private:
  int device;
  int frame_rate;
  int width;
  int height;

  cv::Mat distortion_coefficients_left;
  cv::Mat distortion_coefficients_right;
  cv::Mat camera_matrix_left;
  cv::Mat camera_matrix_right;

  cv::VideoCapture cap;
};

CameraDriver::CameraDriver()
{
  this->camera_matrix_left            = cv::Mat::eye( 3, 3, CV_32FC1 );
  this->camera_matrix_right           = cv::Mat::eye( 3, 3, CV_32FC1 );
  this->distortion_coefficients_left  = cv::Mat::zeros( 1, 5, CV_32FC1 );
  this->distortion_coefficients_right = cv::Mat::zeros( 1, 5, CV_32FC1 );

  this->loadConfig( "/home/lin/Projects/super-vio/config/camera.json" );
  this->initialize();
}

CameraDriver::~CameraDriver()
{
  if ( this->cap.isOpened() )
  {
    this->close();
  }
}

void CameraDriver::loadConfig( const std::string& config_file_path )
{
  std::ifstream  config_file( config_file_path );
  nlohmann::json config;
  config_file >> config;
  config_file.close();

  this->device     = config[ "device" ];
  this->frame_rate = config[ "frame_rate" ];
  this->width      = config[ "resolution" ][ "width" ];
  this->height     = config[ "resolution" ][ "height" ];

  std::cout << "device: " << this->device << std::endl;
  std::cout << "frame_rate: " << this->frame_rate << std::endl;
  std::cout << "width: " << this->width << std::endl;
  std::cout << "height: " << this->height << std::endl;

  double f_x = config[ "camera_matrix" ][ "left" ][ "fx" ];
  double f_y = config[ "camera_matrix" ][ "left" ][ "fy" ];
  double c_x = config[ "camera_matrix" ][ "left" ][ "cx" ];
  double c_y = config[ "camera_matrix" ][ "left" ][ "cy" ];

  this->camera_matrix_left.at<float>( 0, 0 ) = f_x;
  this->camera_matrix_left.at<float>( 1, 1 ) = f_y;
  this->camera_matrix_left.at<float>( 0, 2 ) = c_x;
  this->camera_matrix_left.at<float>( 1, 2 ) = c_y;

  f_x = config[ "camera_matrix" ][ "right" ][ "fx" ];
  f_y = config[ "camera_matrix" ][ "right" ][ "fy" ];
  c_x = config[ "camera_matrix" ][ "right" ][ "cx" ];
  c_y = config[ "camera_matrix" ][ "right" ][ "cy" ];

  this->camera_matrix_right.at<float>( 0, 0 ) = f_x;
  this->camera_matrix_right.at<float>( 1, 1 ) = f_y;
  this->camera_matrix_right.at<float>( 0, 2 ) = c_x;
  this->camera_matrix_right.at<float>( 1, 2 ) = c_y;


  double k1 = config[ "distortion_coefficients" ][ "left" ][ "k1" ];
  double k2 = config[ "distortion_coefficients" ][ "left" ][ "k2" ];
  double p1 = config[ "distortion_coefficients" ][ "left" ][ "p1" ];
  double p2 = config[ "distortion_coefficients" ][ "left" ][ "p2" ];
  double k3 = config[ "distortion_coefficients" ][ "left" ][ "k3" ];

  this->distortion_coefficients_left.at<float>( 0, 0 ) = k1;
  this->distortion_coefficients_left.at<float>( 0, 1 ) = k2;
  this->distortion_coefficients_left.at<float>( 0, 2 ) = p1;
  this->distortion_coefficients_left.at<float>( 0, 3 ) = p2;
  this->distortion_coefficients_left.at<float>( 0, 4 ) = k3;

  k1 = config[ "distortion_coefficients" ][ "right" ][ "k1" ];
  k2 = config[ "distortion_coefficients" ][ "right" ][ "k2" ];
  p1 = config[ "distortion_coefficients" ][ "right" ][ "p1" ];
  p2 = config[ "distortion_coefficients" ][ "right" ][ "p2" ];
  k3 = config[ "distortion_coefficients" ][ "right" ][ "k3" ];

  this->distortion_coefficients_right.at<float>( 0, 0 ) = k1;
  this->distortion_coefficients_right.at<float>( 0, 1 ) = k2;
  this->distortion_coefficients_right.at<float>( 0, 2 ) = p1;
  this->distortion_coefficients_right.at<float>( 0, 3 ) = p2;
  this->distortion_coefficients_right.at<float>( 0, 4 ) = k3;

  std::cout << "distortion_coefficients_left: " << this->distortion_coefficients_left << std::endl;
  std::cout << "distortion_coefficients_right: " << this->distortion_coefficients_right << std::endl;
  std::cout << "camera_matrix_left: \n"
            << this->camera_matrix_left << std::endl;
  std::cout << "camera_matrix_right: \n"
            << this->camera_matrix_right << std::endl;
}

void CameraDriver::initialize()
{
  this->cap.open( this->device );

  if ( !this->cap.isOpened() )
  {
    std::cout << "Error opening video stream or file" << std::endl;
    exit( -1 );
  }

  this->cap.set( cv::CAP_PROP_FPS, this->frame_rate );
  this->cap.set( cv::CAP_PROP_FRAME_WIDTH, this->width * 2 );
  this->cap.set( cv::CAP_PROP_FRAME_HEIGHT, this->height );

  this->cap.set( cv::CAP_PROP_BUFFERSIZE, 1 );
  this->cap.set( cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ) );
}

void CameraDriver::open( const int& device )
{
}

void CameraDriver::close()
{
  this->cap.release();
}

void CameraDriver::getFrame( cv::Mat& frame )
{
  this->cap >> frame;
}

void CameraDriver::getStereoFrameUndistorted( cv::Mat& frame_left, cv::Mat& frame_right )
{
  cv::Mat temp_frame_left, temp_frame_right;

  this->getStereoFrame( temp_frame_left, temp_frame_right );

  cv::undistort( temp_frame_left, frame_left, this->camera_matrix_left, this->distortion_coefficients_left );
  cv::undistort( temp_frame_right, frame_right, this->camera_matrix_right, this->distortion_coefficients_right );
}

void CameraDriver::getStereoFrame( cv::Mat& frame_left, cv::Mat& frame_right )
{
  cv::Mat frame;
  this->cap >> frame;

  frame_left  = frame( cv::Rect( 0, 0, this->width, this->height ) );
  frame_right = frame( cv::Rect( this->width, 0, this->width, this->height ) );
}
