#include "descriptor/cluster.h"

Eigen::MatrixXi sortRowsByL2Norm(Eigen::MatrixXf keys_scan, Eigen::MatrixXf keys_map) {
    int nn_scan = keys_scan.rows();
    int nn_cnt = keys_map.rows();
    Eigen::MatrixXi queue = Eigen::MatrixXi::Zero(nn_scan, nn_cnt);
    for (int i = 0; i < nn_scan; ++i) {
        Eigen::Matrix<float, Eigen::Dynamic, 1> dist = (keys_map.rowwise() - keys_scan.row(i)).rowwise().norm();
        std::vector<std::pair<float, int>> residual (nn_cnt, std::make_pair(0.0, 0));
        for (int j = 0; j < nn_cnt; ++j) {
            residual[j] = std::make_pair(dist(j, 0), j);
        }
        std::sort(residual.begin(), residual.end());
        for (int j = 0; j < nn_cnt; ++j) {
            queue(i, j) = residual[j].second;
        }
    }
    return queue;
}

float distBtnQueuesBySpearman(Eigen::MatrixXi& queue_1, Eigen::MatrixXi& queue_2, const int& n_cols) {
    Eigen::MatrixXi queue_src = queue_1.leftCols(n_cols);
    Eigen::MatrixXi queue_dst = queue_2.leftCols(n_cols);

    std::vector<double> vector_src(queue_src.data(), queue_src.data() + queue_src.size());
    std::vector<double> vector_dst(queue_dst.data(), queue_dst.data() + queue_dst.size());
    return spearman_rank_correlation(vector_src, vector_dst);
}


double spearman_rank_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    // Step 1: Rank the data
    std::vector<int> ranks_x(x.size()), ranks_y(y.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ranks_x[i] = i + 1;
        ranks_y[i] = i + 1;
    }

    // Step 2: Sort the ranks based on the corresponding values
    std::sort(ranks_x.begin(), ranks_x.end(), [&x](int i, int j) { return x[i - 1] < x[j - 1]; });
    std::sort(ranks_y.begin(), ranks_y.end(), [&y](int i, int j) { return y[i - 1] < y[j - 1]; });

    // Step 3: Calculate the differences and square them
    std::vector<double> differences(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        differences[i] = ranks_x[i] - ranks_y[i];
    }

    // Step 4: Sum up the squared differences
    double sum_squared_differences = 0.0;
    for (double diff : differences) {
        sum_squared_differences += diff * diff;
    }

    // Step 5: Use the formula for Spearman's Rank Correlation Coefficient
    size_t n = x.size();
    double rho = 1.0 - (6.0 * sum_squared_differences) / (n * (n * n - 1));

    // Step 6: return loss
    double loss = 1.0 - std::abs(rho);
    return loss;
}