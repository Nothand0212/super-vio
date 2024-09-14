#pragma once
#include <Eigen/Core>
#include <sophus/se3.hpp>
namespace super_vio
{
class CameraParams
{
public:
  // Camera intrinsic parameters
  double fx;
  double fy;
  double cx;
  double cy;

  // Camera distortion parameters
  double k1;
  double k2;
  double p1;
  double p2;
  double k3;

public:
  CameraParams()
  {
    fx = 0.0;
    fy = 0.0;
    cx = 0.0;
    cy = 0.0;
    k1 = 0.0;
    k2 = 0.0;
    p1 = 0.0;
    p2 = 0.0;
    k3 = 0.0;
  }
};


class Camera
{
private:
  CameraParams params_;

  Sophus::SE3d pose_;

public:
  Camera();
  Camera( const CameraParams& params, const Sophus::SE3d& pose );
  ~Camera();

  const Sophus::SE3d&         getPose() const;
  Eigen::Matrix<double, 3, 4> getPoseMatrix();

  const CameraParams& getParams() const;

  void setPose( const Sophus::SE3d& pose );
  void setPose( const Eigen::Matrix<double, 3, 4>& pose_matrix );
  void setParams( const CameraParams& params );
  void setParams( const cv::Mat& intrinsic, const cv::Mat& distortion );
};

Camera::Camera()
{
  params_ = CameraParams();
  pose_   = Sophus::SE3d();
}

Camera::Camera( const CameraParams& params, const Sophus::SE3d& pose )
{
  params_ = params;
  pose_   = pose;
}

Camera::~Camera()
{
}

const Sophus::SE3d& Camera::getPose() const
{
  return pose_;
}

Eigen::Matrix<double, 3, 4> Camera::getPoseMatrix()
{
  //   Eigen::Matrix<double, 3, 4> pose_matrix;
  //   pose_matrix << pose_.rotationMatrix(), pose_.translation();
  //   return pose_matrix;
  return pose_.matrix3x4();
}

const CameraParams& Camera::getParams() const
{
  return params_;
}

void Camera::setPose( const Sophus::SE3d& pose )
{
  pose_ = pose;
}

void Camera::setPose( const Eigen::Matrix<double, 3, 4>& pose_matrix )
{
  pose_ = Sophus::SE3d( pose_matrix.block<3, 3>( 0, 0 ), pose_matrix.block<3, 1>( 0, 3 ) );
}

void Camera::setParams( const CameraParams& params )
{
  params_ = params;
}

void Camera::setParams( const cv::Mat& intrinsic, const cv::Mat& distortion )
{
  if ( intrinsic.size() != cv::Size( 3, 3 ) || distortion.size() != cv::Size( 5, 1 ) )
  {
    std::cerr << "Intrinsic Size: " << intrinsic.size() << " Distortion Size: " << distortion.size() << std::endl;
    std::cerr << "Invalid camera parameters" << std::endl;
    return;
  }

  int type = intrinsic.type();
  if ( type == CV_32F )
  {
    params_.fx = intrinsic.at<float>( 0, 0 );
    params_.fy = intrinsic.at<float>( 1, 1 );
    params_.cx = intrinsic.at<float>( 0, 2 );
    params_.cy = intrinsic.at<float>( 1, 2 );
    params_.k1 = distortion.at<float>( 0, 0 );
    params_.k2 = distortion.at<float>( 0, 1 );
    params_.p1 = distortion.at<float>( 0, 2 );
    params_.p2 = distortion.at<float>( 0, 3 );
    params_.k3 = distortion.at<float>( 0, 4 );
  }
  else if ( type == CV_64F )
  {
    params_.fx = intrinsic.at<double>( 0, 0 );
    params_.fy = intrinsic.at<double>( 1, 1 );
    params_.cx = intrinsic.at<double>( 0, 2 );
    params_.cy = intrinsic.at<double>( 1, 2 );
    params_.k1 = distortion.at<double>( 0, 0 );
    params_.k2 = distortion.at<double>( 0, 1 );
    params_.p1 = distortion.at<double>( 0, 2 );
    params_.p2 = distortion.at<double>( 0, 3 );
    params_.k3 = distortion.at<double>( 0, 4 );
  }
}


}  // namespace super_vio