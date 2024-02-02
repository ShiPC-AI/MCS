#include <pcl/common/io.h>
#include "descriptor/ocsc.h"
#include "base/base.h"
#include "base/io.h"
#include "descriptor/cluster.h"
/* 
** MCS-BF is one-phase place recognition pipeline, which only employs the second-phase search
** It achieves better accuracy than MCS
** 2-phase search (only): OcSC descriptor + occupany loss 
*/
int main(int argc, char** argv) {
    std::cout << "====Online Place Recognition: MCS-BF Recall Top 1====\n";
    std::cout << "We test the overall sequence, this takes a few seconds.\n";

    // dataset and sequence
    std::string dataset = "KITTI";
    std::string seq = "06"; 
    int nn_scan = 1101;

    // directory and path
    std::string dir_project = "../data/" + dataset + "/" + seq;
    std::string dir_data = "/media/spcomen21/WD5T/Dataset/KITTI/";
    std::string dir_pcd = dir_data + seq + "/pcd/";
    std::cout << "Data name: " << dataset << ", " << seq << "\n";

    // parameter
    std::string str_dd = "3"; // string for distance sampling
    float z_min = -0.8, z_max = 2.2; // removing high and low pts
    int num_search = 50, min_move = 50; // num for search loops
    int row = 20, col = 60; // rows, cols for OcSC
    float len = 4.0; std::string str_len = "4"; // rows, cols for OcSC

    std::cout << "=== load gt loop flag nn ===\n";
    Eigen::MatrixXi flag_nn = Eigen::MatrixXi::Zero(nn_scan, nn_scan);
    if (!loadDynamicLoopFlag(dir_project + "/gt/loop_nn.txt", flag_nn)) {
        std::cout << "Load loop closure flag faild.\n";
        return -1;
    }
    Eigen::MatrixXi flag_n1 = flag_nn.rowwise().any();

    OcSCManager ocsc_manager;
    pcl::PointXYZ pt_cnt(0.0, 0.0, 0.0);
    ocsc_manager.setCellParas(row, col, len);
    ocsc_manager.setPassParas(z_min, z_max);
    ocsc_manager.setCenterPt(pt_cnt);

    std::cout << "=== compute recall top 1===\n";
    int tps = 0, fns = 0;
    double time_ocsc = 0.0, time_cluster = 0.0, time_search = 0.0; 
    int count_search = 0, count_ocsc = 0, count_scan = 0;
    Eigen::MatrixXf time_scan((int)flag_n1.sum(), 3);

    std::clock_t tstart_sequence = std::clock();
    std::vector<Eigen::MatrixXi> ocscs_scan(nn_scan);
    for (int i = 0; i < nn_scan; ++i) {
        printInfo(i, 500, "pcd");
        CloudPtr cloud_scan(new PointCloud);
        pcl::io::loadPCDFile(dir_pcd + std::to_string(i) + ".pcd", *cloud_scan);
        
        // time descriptor
        std::clock_t tstart_ocsc = std::clock();
        ocsc_manager.loadRawCloud(cloud_scan, 8000);
        ocscs_scan[i] = ocsc_manager.makeOcSC();
        std::clock_t tend_ocsc = std::clock();
        
        time_ocsc += durationTime(tstart_ocsc, tend_ocsc);//ms
        ++count_ocsc;

        // jump non-loop
        if (flag_n1(i, 0) < 1) {
            continue;
        }

        // Brute force search
        std::clock_t tstart_search = std::clock();
        std::vector<float> residual;
        for (int j = 0; j < i; ++j) {
            if ((i - j) < min_move) {
                continue;
            }
            float loss = distBtnOcSCsByWeightedRatioandNum(ocscs_scan[i], ocscs_scan[j], 0.85, 0.15);
            residual.push_back(loss);
        }
        auto min_element = std::min_element(residual.begin(), residual.end());
        int id_find = min_element - residual.begin();

        std::clock_t tend_search = std::clock();
        time_search += durationTime(tstart_search, tend_search); //ms
        ++count_search;

        if (flag_nn(i, id_find) > 0) {
            ++tps;
        } else {
            ++fns;
        }

        time_scan.row(count_scan) << i, time_ocsc, time_search;
        ++count_scan;
    }
    std::clock_t tend_sequence = std::clock();
    double time_sequence = (double)durationTime(tstart_sequence, tend_sequence);
    
    std::cout << "Loop closure detection recall: " << (float)tps / (tps + fns) << "\n"; 
    std::cout << "Creating OcSC time: " << (float)time_ocsc/count_ocsc << "ms\n";
    std::cout << "Nearest neighbor search time: " << (float)time_search/count_search << "ms\n";
    std::cout << "Overall sequence time: " << time_sequence << "ms\n";

    return 1;
}