#ifndef _MCS_BASE_BASE_H
#define _MCS_BASE_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "base/utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void randomSampleCloud(const CloudPtr& cloud_in, CloudPtr& cloud_out,  int N);

void voxelSampleCloud(const CloudPtr& cloud_in, CloudPtr& cloud_out, 
    const float leaf_x, const float leaf_y, const float leaf_z);

void passThrough(const CloudPtr& cloud_in, CloudPtr& cloud_out, 
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    const float min_z, const float max_z, bool is_negative);
    

double durationTime(std::clock_t& start_time, std::clock_t& end_time);

template <typename T>
cv::Mat mat2Image(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    cv::Mat image(rows, cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            image.at<uchar>(i, j) = (int)(mat(i, j) * 254);
        }
    }
    return (image.clone());
}
#endif//MCS_BASE_BASE_H