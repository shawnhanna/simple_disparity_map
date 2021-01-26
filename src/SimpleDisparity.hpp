#pragma once

#include <opencv2/core.hpp>

struct SimpleDisparityParams
{
    int box_width = 15;
    int box_height = 15;
    int max_disparity_pixels = 60;
    int resize_factor = 1;
};

class SimpleDisparity
{
private:
    SimpleDisparityParams params_;

    cv::Mat img_left_, img_right_;

    int computeSAD(int x_1, int y_1, int x_2, int y_2);
    int findMinCostShift(int in_x_, int y_);

    /* data */
public:
    SimpleDisparity(/* args */){};
    ~SimpleDisparity(){};

    void setParams(const SimpleDisparityParams& params){
        params_ = params;
    }

    bool computeDisparity(const cv::Mat &img_left, const cv::Mat &img_right, cv::Mat &img_out);
};
