#include "image_process.h"

cv::Mat normalizeImage( const cv::Mat& image )
{
  cv::Mat image_normalized = image.clone();

  if ( image_normalized.channels() == 3 )
  {
    cv::cvtColor( image_normalized, image_normalized, cv::COLOR_BGR2RGB );
    image_normalized.convertTo( image_normalized, CV_32FC3, 1.0f / 255.0f );
  }
  else if ( image_normalized.channels() == 1 )
  {
    image_normalized.convertTo( image_normalized, CV_32FC1, 1.0f / 255.0f );
  }
  else
  {
    throw std::runtime_error( "Unsupported image type." );
  }

  return image_normalized;
}


std::vector<cv::Point2f> normalizeKeyPoints( std::vector<cv::Point2f> key_points, const int& height, const int& width )
{
  cv::Size size( width, height );

  cv::Point2f shift( static_cast<float>( size.width ) / 2.0f, static_cast<float>( size.height ) / 2.0f );

  float scale = static_cast<float>( std::max( width, height ) ) / 2;

  std::vector<cv::Point2f> key_points_normalized;

  for ( const cv::Point2f& key_point : key_points )
  {
    cv::Point2f key_point_normalized = ( key_point - shift ) / scale;
    key_points_normalized.push_back( key_point_normalized );
  }
  return key_points_normalized;
}

cv::Mat resizeImage( const cv::Mat& image, const int& size, float& scale, const std::string& fn, const std::string& interpolation )
{
  int width  = image.cols;
  int height = image.rows;

  std::function<int( int, int )> func;

  if ( fn == "max" )
  {
    func = []( int a, int b ) { return std::max( a, b ); };
  }
  else if ( fn == "min" )
  {
    func = []( int a, int b ) { return std::min( a, b ); };
  }
  else
  {
    throw std::runtime_error( "Unsupported function." );
  }

  int width_new, height_new;
  if ( size == 512 || size == 1024 || size == 2048 )
  {
    scale      = static_cast<float>( size ) / static_cast<float>( func( width, height ) );
    width_new  = static_cast<int>( round( width * scale ) );
    height_new = static_cast<int>( round( height * scale ) );
  }
  else
  {
    throw std::invalid_argument( "Unsupported size: " + std::to_string( size ) );
  }

  cv::InterpolationFlags mode;
  if ( interpolation == "linear" )
  {
    mode = cv::INTER_LINEAR;
  }
  else if ( interpolation == "cubic" )
  {
    mode = cv::INTER_CUBIC;
  }
  else if ( interpolation == "nearest" )
  {
    mode = cv::INTER_NEAREST;
  }
  else if ( interpolation == "area" )
  {
    mode = cv::INTER_AREA;
  }
  else
  {
    throw std::invalid_argument( "[ERROR] Incorrect interpolation mode: " + interpolation );
  }

  cv::Mat image_resized;
  cv::resize( image, image_resized, cv::Size( width_new, height_new ), 0, 0, mode );

  return image_resized;
}
