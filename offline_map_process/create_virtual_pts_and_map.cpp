#include <cstdlib>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include "ground/ground_segmentation.h"
#include "base/base.h"
#include "base/io.h"
/* 
** create a map point cloud and virtual points from a set of offline invidual scans
** input: (1) ground-truth poses (or odometry pose), (2) multiple LiDAR scans
** output: (1) map point cloud, (2) virtual points 
*/
int main (int argc, char** argv) {
    std::cout << "====Offline Map Process: Create Virtual Points and Map Point Cloud====\n";
    std::cout << "This may take a few seconds, please wait...\n";

    // dataset and sequence
    std::string dataset = "KITTI";
    std::string seq = "06"; 
    int nn_scan = 1101;

    // directory and path
    std::string dir_project = "../data/" + dataset + "/" + seq;
    std::string dir_map = dir_project + "/map/";
    std::string dir_data = "/media/spcomen21/WD5T/Dataset/KITTI/";
    std::string dir_pcd = dir_data + seq + "/pcd/";
    std::cout << "Data name: " << dataset << ", " << seq << "\n";
    
    // parameter
    float z_min = -0.8, z_max = 2.2; // removing high and low pts
    float dd_sample = 1.5; // distance for keyframe 
    int nn_sam_ground = 10000; // sampling num for ground
    int nn_sam_map = 100000; // sampling num for map
    float leaf_size = 3.0; // voxel leaf for ground

    // load pose && extrat xyz
    Eigen::MatrixXf poses_gt(nn_scan, 16);
    if (!loadDynamicMatrix(dir_project + "/gt/gt_poses.txt", poses_gt)) {
        std::cout << "Load gt pose failed.\n";
        return -1;
    }
    Eigen::MatrixXf poses_xyz = poses_gt(Eigen::all, Eigen::seqN(3, 3, 4));
    std::cout << "Pose Rows: " << poses_gt.rows() << ", Cols: " << poses_gt.cols() << "\n";
    
    // extract keyframe id
    Eigen::MatrixXf xyz_front = poses_xyz.row(0);
    std::vector<int> ids_key{0};
    for (int i = 1; i < nn_scan; ++i) {
        float dist = (xyz_front - poses_xyz.row(i)).norm();
        if (dist < dd_sample) {
            continue;
        }
        xyz_front = poses_xyz.row(i);
        ids_key.push_back(i);
    }
    Eigen::MatrixXf pose_key = poses_gt(ids_key, Eigen::all);
    std::cout << "Keyframe size: " << pose_key.rows() << ", " << pose_key.cols() << "\n";

    // add ground (dense) and map (dense) by keyframes
    CloudPtr ground_map(new PointCloud);
    CloudPtr cloud_map(new PointCloud);

    Eigen::MatrixXf pose_key_trans = pose_key.transpose(); // column first
    for (int i = 0; i < ids_key.size(); ++i) {
        printInfo(i, 30, "ground.");
        
        // load individual scan
        CloudPtr cloud_scan(new PointCloud);
        pcl::io::loadPCDFile(dir_pcd + std::to_string(ids_key[i]) + ".pcd", *cloud_scan);
        
        // remove low and high points (height)
        CloudPtr cloud_pass(new PointCloud);
        passThrough(cloud_scan, cloud_pass, -100, 100, -100, 100, z_min, z_max, false);

        // transform scan by ground-truth pose
        CloudPtr cloud_trans(new PointCloud);
        Eigen::Map<Eigen::MatrixXf> pose_44(pose_key_trans.col(i).data(), 4, 4);
        pcl::transformPointCloud(*cloud_pass, *cloud_trans, pose_44.transpose());
        
        // segment ground from raw individual scan
        GroundSegmentationParams params;
        GroundSegmentation segmenter(params);
        std::vector<int> labels;
        segmenter.segment(*cloud_scan, &labels);
        CloudPtr ground_scan(new PointCloud);
        for (int j = 0; j < labels.size(); ++j) {
            if (labels[j] == 1) {
                ground_scan->push_back(std::move(cloud_scan->points[j]));
            }
        }

        // transform scan ground by ground-truth pose
        CloudPtr ground_trans(new PointCloud);
        pcl::transformPointCloud(*ground_scan, *ground_trans, pose_44.transpose());
        
        // stitch ground and map
        *ground_map += *ground_trans;
        *cloud_map += *cloud_trans;
    }
    std::cout << "Dense Map size: " << cloud_map->size() << "\n";
    std::cout << "Dense Map Ground size: " << ground_map->size() << "\n";

    // down-sample and project the dense map
    CloudPtr cloud_map_sam(new PointCloud);
    randomSampleCloud(cloud_map, cloud_map_sam, nn_sam_map);
    CloudPtr cloud_map_pro(new PointCloud);
    pcl::copyPointCloud(*cloud_map_sam, *cloud_map_pro);
    for (int i = 0; i < cloud_map_pro->size(); ++i) {
        cloud_map_pro->at(i).z = 0;
    }
    pcl::io::savePCDFileBinary(dir_map + "map_sam_pro.pcd", *cloud_map_pro); 
    std::cout << "Pro Map size: " << cloud_map_sam->size() << "\n";

    // project and down-sample the dense ground
    CloudPtr ground_pro(new PointCloud);
    pcl::copyPointCloud(*ground_map, *ground_pro);
    for (int i = 0; i < ground_pro->size(); ++i) {
        ground_pro->at(i).z = 0;
    }
    
    CloudPtr ground_sam(new PointCloud);
    randomSampleCloud(ground_pro, ground_sam, nn_sam_ground);
    // pcl::io::savePCDFileBinary(dir_map + "ground_sam.pcd", *ground_sam);
    std::cout << "Random sampling ground size: " << ground_sam->size() << "\n";

    CloudPtr ground_sam_dist(new PointCloud);
    voxelSampleCloud(ground_sam, ground_sam_dist, leaf_size, leaf_size, leaf_size);
    std::cout << "Distance sampling ground size: " << ground_sam_dist->size() << "\n";
    pcl::io::savePCDFileBinary(dir_map + "ground_sam_dist_3.pcd", *ground_sam_dist);

    return 1;
}