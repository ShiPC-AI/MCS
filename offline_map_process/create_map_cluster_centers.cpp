#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include "descriptor/ocsc.h"
#include "base/base.h"
#include "base/io.h"
#include "offline_map_process/cluster/k_means.h"
/* 
** create map cluster centers from virtual points and map 
** input: (1) virtual points, (2) projected-downsamping map
** output: map cluster centers 
*/
int main(int argc, char** argv) {
    std::cout << "====Offline Map Process: Create Map Cluster Centers====\n";
    std::cout << "This may take a few seconds, please wait...\n";

    // dataset and sequence
    std::string dataset = "KITTI";
    std::string seq = "06"; 
    int nn_scan = 1101;

    // directory and path
    std::string dir_project = "../data/" + dataset + "/" + seq;
    std::string dir_map = dir_project + "/map/";
    std::cout << "Data name: " << dataset << ", " << seq << "\n";

    // parameter
    std::string str_dd = "3"; // string for distance sampling
    float z_min = -0.8, z_max = 2.2; // removing high and low pts

    int nn_sam_ocsc = 8000; // sampling num for OcSC's input
    int row = 20, col = 60; // rows, cols for OcSC
    float len = 4.0; std::string str_len = "4";  // rows, cols for OcSC
    std::string para_ocsc = str_len + "_" + std::to_string(row) + "_" + std::to_string(col);
    std::string para_map = para_ocsc + "_" + str_dd;

    // load projected map
    CloudPtr map_pro(new PointCloud);
    pcl::io::loadPCDFile(dir_map + "map_sam_pro.pcd", *map_pro);
    std::cout << "Projected map size: " << map_pro->size() << "\n";

    // load sampling map ground
    CloudPtr map_ground(new PointCloud);
    pcl::io::loadPCDFile(dir_map + "ground_sam_dist_" + str_dd + ".pcd", *map_ground);
    std::cout << "Virtual pts size: " << map_ground->size() << "\n";
    
    // make ocscs && ringkeys
    int nn_ground = map_ground->points.size();
    Eigen::MatrixXf ring_keys(nn_ground, row); // num of ground * row
    // build tree and indices
    OcSCManager ocsc_manager;
    ocsc_manager.setCellParas(row, col, len);
    ocsc_manager.setPassParas(z_min, z_max);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(map_pro);
    pcl::ExtractIndices<pcl::PointXYZ>::Ptr ids_ext(new pcl::ExtractIndices<pcl::PointXYZ>);
    ids_ext->setInputCloud(map_pro);  

    for (int i = 0; i < map_ground->points.size(); ++i) {
        printInfo(i, 500, "pt");
        std::vector<int> ids;
        std::vector<float> dists;
        pcl::PointXYZ pt = map_ground->points[i];
        tree->radiusSearch(pt, (float)row * len, ids, dists);

        // find neighbour map points  
        CloudPtr cloud_around(new PointCloud);
        pcl::IndicesPtr ids_ptr = std::make_shared<std::vector<int>>(ids);
        ids_ext->setIndices(ids_ptr);
        ids_ext->setNegative(false);
        ids_ext->filter(*cloud_around);
        
        // make ocsc and ringkeys
        ocsc_manager.setCenterPt(pt);
        ocsc_manager.loadRawCloud(cloud_around, nn_sam_ocsc); 
        Eigen::MatrixXi ocsc = ocsc_manager.makeOcSC();
        Eigen::MatrixXf ring_key = ocsc_manager.getOcSCRingKey();
        ring_keys.row(i) = ring_key.transpose();
    }
    std::cout << "All ring keys together dimension: " << ring_keys.rows() << ", " << ring_keys.cols() << "\n";
    
    // save map cluster centers
    saveDynamicMatrix(dir_map + "map_ring_keys_" + para_map + ".txt", ring_keys);
    
    // k_means cluster (C++ version)
    int num_cluster = 50;
    int max_iteration = 50;
    Eigen::MatrixXf cluster_cnts = kMeansClustering(ring_keys, num_cluster, max_iteration);

    std::cout << "---Save map cluster centers into binary files, occupying only 4 Kb---\n"; 
    // saveDynamicMatrix(dir_map + "/cnts_" + para_map + ".txt", cluster_cnts); // ascii
    if(!saveDynamicMatrixBinary(dir_map + "/cnts_" + para_map + ".bin", cluster_cnts)) { //binary
        return -1;
    }

    return 0;
}
