#pragma once

#include <opencv2/opencv.hpp>

namespace super_vio
{
class Region
{
public:
  cv::Rect2i      rectangle;
  std::deque<int> indexs;
};

class Features
{
private:
  std::vector<cv::Point2f> m_v_key_points;
  std::vector<float>       m_v_scores;
  cv::Mat                  m_mat_descriptor;

public:
  Features()  = default;
  ~Features() = default;

  void operator=( const Features& features );

  void setKeyPoints( const std::vector<cv::Point2f>& key_points );


  void setScores( const std::vector<float>& scores );

  void setDescriptor( const cv::Mat& descriptor );

  std::vector<cv::Point2f> getKeyPoints() const;

  std::vector<float> getScores() const;

  cv::Mat getDescriptor() const;
};
}  // namespace super_vio