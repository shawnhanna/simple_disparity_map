
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "SimpleDisparity.hpp"

bool readImage(std::string &filename, cv::Mat &mat)
{
  // read image
  mat = cv::imread(filename);
  if (mat.empty()) {
    std::cerr << "Failed to read file: " << filename << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char const *argv[])
{
  std::string img_left_name, img_right_name, img_out_name;

  // read args
  if (argc == 3){
    img_left_name = std::string(argv[1]);
    img_right_name = std::string(argv[2]);
  }
  else if (argc == 4){
    // input images
    img_left_name = std::string(argv[1]);
    img_right_name = std::string(argv[2]);
    // output image
    img_out_name = std::string(argv[3]);
  }
  else {
    std::cerr << "Invalid args. Need: ./simple_disparity_map img_left.png img_right.png [img_out.png]" << std::endl;
    return 100;
  }

  std::cout << "Reading images: " << img_left_name <<", " << img_right_name << std::endl;
  cv::Mat img_left, img_right, img_out;
  if (!readImage(img_left_name, img_left))
    return 1;
  if (!readImage(img_right_name, img_right))
    return 1;

  std::cout << "computing disparity" << std::endl;
  if (img_out_name.length() == 0){
    cv::imshow("img left", img_left);
    cv::imshow("img right", img_right);
    cv::waitKey(100);
  }

  SimpleDisparity disparity;
  disparity.computeDisparity(img_left, img_right, img_out);
  if (img_out_name.length() == 0){
    std::cout << "Press 'q' to quit" << std::endl;
    cv::imshow("output", img_out);
    while(cv::waitKey(-1) != 'q'){}
  }
  else{
    std::cout << "Writing output: " << img_out_name << std::endl;
    cv::imwrite(img_out_name, img_out);
  }

  return 0;
}
