#pragma once


#include <opencv2/opencv.hpp>

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

  void operator=( const Features& features )
  {
    m_v_key_points   = features.m_v_key_points;
    m_v_scores       = features.m_v_scores;
    m_mat_descriptor = features.m_mat_descriptor.clone();
  }

  void setKeyPoints( const std::vector<cv::Point2f>& key_points )
  {
    m_v_key_points = key_points;
  }


  void setScores( const std::vector<float>& scores )
  {
    m_v_scores = scores;
  }

  void setDescriptor( const cv::Mat& descriptor )
  {
    m_mat_descriptor = descriptor.clone();
  }

  std::vector<cv::Point2f> getKeyPoints() const
  {
    return m_v_key_points;
  }

  std::vector<float> getScores() const
  {
    return m_v_scores;
  }

  cv::Mat getDescriptor() const
  {
    return m_mat_descriptor;
  }
};