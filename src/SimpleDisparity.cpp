#include "SimpleDisparity.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <opencv2/imgproc.hpp>

/**
 * returns true if the given x/y coordinate fall inside an image bounds
 */
bool inBounds(const cv::Mat &mat, int x, int y){
  if (x < 0 || x >= mat.cols)
    return false;

  if (y < 0 || y >= mat.rows)
    return false;

  return true;
}

/**
 * calculates the abs difference between two blocks given the top left
 *  corner of each ROI
 * 
 * x1/y1 = center pixel to search in left image
 * x2/y2 = center pixel to search in right image
 */
int SimpleDisparity::computeSAD(int x_1, int y_1, int x_2, int y_2)
{
  // std::cout << cv::Point(x_1, y_1) <<" - " << cv::Point(x_2, y_2)<< std::endl;
  int sum_sq = 0;
  for (int x = -params_.box_width/2; x < params_.box_width/2; ++x){
    for (int y = -params_.box_height/2; y < params_.box_height/2; ++y){
      cv::Point p1(x_1 + x, y_1 + y);
      cv::Point p2(x_2 + x, y_2 + y);
      if (!(inBounds(img_left_, p1.x, p1.y) && inBounds(img_right_, p2.x, p2.y))){
        continue;
      }
      auto v1 = img_left_.at<uchar>(p1);
      auto v2 = img_right_.at<uchar>(p2);
      sum_sq += abs(v2 - v1);
      // std::cout << (int)v1 <<',' << (int)v2 << " = " << sum_sq << std::endl;
    }
  }
  return sum_sq;
}

/**
 * Find the cost at the given pixel. Slides to the right and finds the min cost
 *  pixel shift amount
 */
int SimpleDisparity::findMinCostShift(int in_x_, int in_y_)
{
  if (!inBounds(img_left_, in_x_, in_y_)){
    return -1;
  }

  // search to the right of the pixel and pick the smallest box diff
  std::pair<int, int> min_cost(999999999, -1);
  for (int x = in_x_; x < in_x_ + params_.max_disparity_pixels; ++x)
  {
    if(!inBounds(img_left_, x, in_y_))
      continue;

    int cost = computeSAD(x, in_y_, in_x_, in_y_);

    if (cost < min_cost.first){
      min_cost.first = cost;
      min_cost.second = x;
    }
  }
  if (min_cost.second == -1)
    return -1;

  return abs(min_cost.second - in_x_);
}

/**
 * computes the disparity between two images, and outputs it as a CV_8U output image.
 *    Normalizes from 0->255 in the output
 */
bool SimpleDisparity::computeDisparity(
            const cv::Mat &img_left,
            const cv::Mat &img_right,
            cv::Mat &img_out)
{
  if (img_left.size() != img_right.size()){
    std::cerr << "Cannot compute disparity on different sized images" << std::endl;
    return false;
  }

  // convert to grayscale
  cv::cvtColor(img_left, img_left_, cv::COLOR_RGB2GRAY, CV_8U);
  cv::cvtColor(img_right, img_right_, cv::COLOR_RGB2GRAY, CV_8U);

  // resize if needed
  if (params_.resize_factor != 1)
  {
    std::cout << "Resizing: scale = " << params_.resize_factor << std::endl;
    cv::resize(img_left_, img_left_, img_left_.size() / params_.resize_factor);
    cv::resize(img_right_, img_right_, img_right_.size() / params_.resize_factor);
  }

  // create output image
  img_out.create(img_left_.size(), CV_16UC1);

  // variables for percent complete printing
  int completed = 0;
  int last_percent = -10;
  // multithread the operation
  #pragma omp parallel for
  for (int x=0;x < img_left_.cols; ++x)
  {
    for (int y=0; y < img_left_.rows; ++y)
    {
      int shift = findMinCostShift(x, y);

      if (shift >= 0)
      {
        img_out.at<uint16_t>(cv::Point(x, y)) = (uint16_t)shift;
      }
    }

    // print percent complete
    #pragma omp critical
    {
      ++completed;
      int percent = 100 * double(completed) / (img_left_.cols);
      if (last_percent != percent)
      {
        std::cout << percent << " percent done" << std::endl;
        last_percent = percent;
      }
    }
  }
  cv::normalize(img_out, img_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}
