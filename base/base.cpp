#include "base/base.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

void randomSampleCloud(const CloudPtr& cloud_in, 
    CloudPtr& cloud_out, int N) {
    pcl::RandomSample<pcl::PointXYZ> random_sampler;
    random_sampler.setInputCloud(cloud_in);
    random_sampler.setSample(N);
    random_sampler.filter(*cloud_out);
}

void voxelSampleCloud(const CloudPtr& cloud_in, CloudPtr& cloud_out, 
    const float leaf_x, const float leaf_y, const float leaf_z) {
    pcl::VoxelGrid<pcl::PointXYZ> grid_voxel;
    grid_voxel.setInputCloud(cloud_in);
    grid_voxel.setLeafSize(leaf_x, leaf_y, leaf_z);
    grid_voxel.filter(*cloud_out);
}

void passThrough(const CloudPtr& cloud_in, CloudPtr& cloud_out, 
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    const float min_z, const float max_z, bool is_negative ){
    
    bool use_x = min_x < max_x;
    bool use_y = min_y < max_y;
    bool use_z = min_z < max_z;
    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices());
    for (int i = 0; i < cloud_in->size(); ++i) {
        const pcl::PointXYZ& pt = cloud_in->points[i];
        if (use_x && (pt.x < min_x || pt.x > max_x)) {
            continue;
        }
        if (use_y && (pt.y < min_y || pt.y > max_y)) {
            continue;
        }
        if (use_z && (pt.z < min_z || pt.z > max_z)) {
            continue;
        }
        inliers_ptr->indices.push_back(i);
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers_ptr);
    extract.setNegative(is_negative);
    extract.filter(*cloud_out);
    return;
}



// ms
double durationTime(std::clock_t& start_time, std::clock_t& end_time) {
    
    return double(end_time - start_time) * MS_SEC;
}
