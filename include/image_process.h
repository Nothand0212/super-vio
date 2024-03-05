/**
 ******************************************************************************
 * @file           : include/image_process.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat normalizeImage( const cv::Mat& image );

std::vector<cv::Point2f> normalizeKeyPoints( std::vector<cv::Point2f> key_points, const int& height, const int& width );

cv::Mat resizeImage( const cv::Mat& image, const int& size, float& scale, const std::string& fn = "max", const std::string& interpolation = "linear" );
