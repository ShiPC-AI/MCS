#ifndef _MCS_CLUSTER_K_MEANS_H
#define _MCS_CLUSTER_K_MEANS_H
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "base/base.h"
#include "base/io.h"
struct Vector {
    Eigen::VectorXf values;
};

struct Cluster {
    Vector centroid;
    std::vector<Vector> vectors;
};

float calculateDistance(const Vector& v1, const Vector& v2);

void assignVectorsToClusters(const Eigen::MatrixXf& data, std::vector<Cluster>& clusters);

void updateClusterCentroids(std::vector<Cluster>& clusters);

Eigen::MatrixXf kMeansClustering(Eigen::MatrixXf& data, int k, int maxIterations);
#endif//MCS_CLUSTER_K_MEANS_H