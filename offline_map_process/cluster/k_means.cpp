#include "offline_map_process/cluster/k_means.h"

float calculateDistance(const Vector& v1, const Vector& v2) {
    return (v1.values - v2.values).squaredNorm();
}

void assignVectorsToClusters(const Eigen::MatrixXf& data, std::vector<Cluster>& clusters) {
    for (auto& cluster : clusters) {
        cluster.vectors.clear();
    }

    for (int i = 0; i < data.rows(); ++i) {
        float minDistance = std::numeric_limits<float>::max();
        int closestCluster = 0;

        for (int j = 0; j < clusters.size(); ++j) {
            float distance = calculateDistance({data.row(i)}, clusters[j].centroid);
            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = j;
            }
        }

        clusters[closestCluster].vectors.push_back({data.row(i)});
    }
}

void updateClusterCentroids(std::vector<Cluster>& clusters) {
    for (auto& cluster : clusters) {
        if (!cluster.vectors.empty()) {
            int dimension = cluster.vectors[0].values.size();
            cluster.centroid.values.setZero();

            for (const auto& vector : cluster.vectors) {
                cluster.centroid.values += vector.values;
            }

            cluster.centroid.values /= cluster.vectors.size();
        }
    }
}

Eigen::MatrixXf kMeansClustering(Eigen::MatrixXf& data, int k, int maxIterations) {
    std::vector<Cluster> clusters;
    std::srand(std::time(0));

    // 随机选择簇的中心点
    for (int i = 0; i < k; ++i) {
        int randomIndex = std::rand() % data.rows();
        Cluster cluster;
        cluster.centroid = {data.row(randomIndex)};
        clusters.push_back(cluster);
    }

    // 迭代更新簇的中心点
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        assignVectorsToClusters(data, clusters);
        updateClusterCentroids(clusters);
    }

    // 返回最终的簇中心点
    Eigen::MatrixXf centroids(k, data.cols());
    for (int i = 0; i < k; ++i) {
        centroids.row(i) = clusters[i].centroid.values;
    }

    return centroids;
}