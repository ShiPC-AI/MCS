#ifndef _MCS_DESCRIPTOR_CLUSTER_H
#define _MCS_DESCRIPTOR_CLUSTER_H
#include "base/base.h"

// compute cluster queue descriptor
Eigen::MatrixXi sortRowsByL2Norm(Eigen::MatrixXf keys_scan, Eigen::MatrixXf keys_map);

float distBtnQueuesBySpearman(Eigen::MatrixXi& queue_1, Eigen::MatrixXi& queue_2, const int& n_cols);
double spearman_rank_correlation(const std::vector<double>& x, const std::vector<double>& y);

#endif //_MCS_DESCRIPTOR_CLUSTER_H