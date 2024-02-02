#ifndef _MCS_BASE_UTILS_H
#define _MCS_BASE_UTILS_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudRGBPtr;

typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointCloud<pcl::Normal>::Ptr NormalPtr;
const double MS_SEC = (double)1000 / CLOCKS_PER_SEC;

typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Mat44Vec;

#endif // _MCS_BASE_UTILS_HPP