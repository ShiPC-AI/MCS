#include <pcl/common/io.h>
#include "descriptor/ocsc.h"
#include "base/base.h"
#include "base/io.h"
#include "descriptor/cluster.h"
/* 
** MCS is two-phase place recognition pipeline, which balances runtime with accuracy
** 1-phase: cluster descriptor + clustering Separman loss 
** 2-phase: OcSC descriptor + occupany loss 
*/
int main(int argc, char** argv) {
    std::cout << "====Online Place Recognition: MCS Recall Top 1====\n";
    std::cout << "We test the overall sequence, this takes a few seconds.\n";

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
    std::string str_dd = "3"; // string for distance sampling
    float z_min = -0.8, z_max = 2.2; // removing high and low pts
    int num_search = 50, min_move = 50; // num for search loops
    int nn_cnts = 50; // num for map clusters
    int row = 20, col = 60; // rows, cols for OcSC
    float len = 4.0; std::string str_len = "4"; // rows, cols for OcSC
    std::string para_ocsc = str_len + "_" + std::to_string(row) + "_" + std::to_string(col);
    std::string para_map = para_ocsc + "_" + str_dd;

    std::cout << "=== load gt loop flag nn ===\n";
    Eigen::MatrixXi flag_nn = Eigen::MatrixXi::Zero(nn_scan, nn_scan);
    if (!loadDynamicLoopFlag(dir_project + "/gt/loop_nn.txt", flag_nn)) {
        std::cout << "Load loop closure flag faild.\n";
        return -1;
    }
    Eigen::MatrixXi flag_n1 = flag_nn.rowwise().any();

    std::cout << "=== load map cluster centers ===\n"; 
    Eigen::MatrixXf cluster_cnts(nn_cnts, row);
    if (!loadDynamicMatrixBinary(dir_map + "/cnts_" + para_map + ".bin", cluster_cnts)) {
    // if (!loadDynamicMatrix(dir_map + "/cnts_" + para_map + ".txt", cluster_cnts)) {
        std::cout << "Load map clusters centers failed.\n";
        return -1;
    }

    OcSCManager ocsc_manager;
    pcl::PointXYZ pt_cnt(0.0, 0.0, 0.0);
    ocsc_manager.setCellParas(row, col, len);
    ocsc_manager.setPassParas(z_min, z_max);
    ocsc_manager.setCenterPt(pt_cnt);

    std::cout << "=== compute recall top 1===\n";
    int tps = 0, fns = 0;
    double time_ocsc = 0.0, time_cluster = 0.0; 
    double time_search_1 = 0.0, time_search_2 = 0.0; 
    int count_search = 0, count_ocsc = 0, count_scan = 0;
    Eigen::MatrixXf time_scan((int)flag_n1.sum(), 5);

    std::clock_t tstart_sequence = std::clock();
    std::vector<Eigen::MatrixXi> ocscs_scan(nn_scan);
    std::vector<Eigen::MatrixXi> clusters_scan(nn_scan);
    for (int i = 0; i < nn_scan; ++i) {
        printInfo(i, 500, "pcd");
        CloudPtr cloud_scan(new PointCloud);
        pcl::io::loadPCDFile(dir_pcd + std::to_string(i) + ".pcd", *cloud_scan);
        
        // time descriptor
        std::clock_t tstart_ocsc = std::clock();
        ocsc_manager.loadRawCloud(cloud_scan, 8000);
        Eigen::MatrixXi ocsc = ocsc_manager.makeOcSC();
        ocscs_scan[i] = ocsc;
        std::clock_t tend_ocsc = std::clock();
        
        time_ocsc += durationTime(tstart_ocsc, tend_ocsc);//ms
        ++count_ocsc;

        // compute key and queue
        std::clock_t tstart_cluster = std::clock();
        Eigen::MatrixXf ring_key_scan = ocsc_manager.getOcSCRingKey();
        clusters_scan[i] = sortRowsByL2Norm(ring_key_scan.transpose(), cluster_cnts);

        std::clock_t tend_cluster = std::clock();
        time_cluster += durationTime(tstart_cluster, tend_cluster);
        
        // jump non-loop
        if (flag_n1(i, 0) < 1) {
            continue;
        }
        
        // 1-phase search
        std::clock_t tstart_search_1 = std::clock();
        std::vector<std::pair<float, int>> losses;
        for (int j = 0; j < i; ++j) {
            if ((i - j) < min_move) {
                continue;
            }
            float loss = distBtnQueuesBySpearman(clusters_scan[i], clusters_scan[j], nn_cnts);
            losses.push_back(std::make_pair(loss, j));
        }
        std::sort(losses.begin(), losses.end());

        std::clock_t tend_search_1 = std::clock();
        time_search_1 += durationTime(tstart_search_1, tend_search_1); //ms

        // 2-phase search
        std::clock_t tstart_search_2 = std::clock();
        int num_bf = std::min(num_search, (int) losses.size());
        std::vector<float> residual(num_bf);
        for (int j = 0; j < num_bf; ++j) {
            int id = losses[j].second;
            float loss = distBtnOcSCsByWeightedRatioandNum(ocscs_scan[i], ocscs_scan[id], 0.85, 0.15);
            residual[j] = loss;
        }
        auto min_element = std::min_element(residual.begin(), residual.end());
        int id_find = losses[min_element - residual.begin()].second;
        
        std::clock_t tend_search_2 = std::clock();
        time_search_2 += durationTime(tstart_search_2, tend_search_2) ; // ms
        ++count_search;

        if (flag_nn(i, id_find) > 0) {
            ++tps;
        } else {
            ++fns;
        }

        time_scan.row(count_scan) << i, time_ocsc, time_cluster, time_search_1, time_search_2;
        ++count_scan;
    }

    std::clock_t tend_sequence = std::clock();
    double time_sequence = durationTime(tstart_sequence, tend_sequence);

    std::cout << "Loop closure detection recall: " << (float)tps / (tps + fns) << "\n"; 
    std::cout << "Creating OcSC time: " << (float)time_ocsc/count_ocsc << "ms\n";
    std::cout << "Search 1 time: " << (float)time_search_1/count_search << "ms\n";
    std::cout << "Search 2 time: " << (float)time_search_2/count_search << "ms\n";
    std::cout << "Overall sequence time: " << time_sequence << "ms\n";

    return 1;
}