#include "descriptor/ocsc.h"

Eigen::MatrixXi OcSCManager::makeOcSC() {
    _ocsc = Eigen::MatrixXi::Zero(_rows, _cols);
    double d_max = _rows * _cell_len;
    double PI_2 = M_PI * 2.0;
    float reso_theta = PI_2 / _cols;

    for (int i = 0; i < _cloud_src->size(); ++i) {
        const pcl::PointXYZ &pt = _cloud_src->points[i];
        if (pt.z > _z_max || pt.z < _z_min) {
            continue;
        }

        float dx = pt.x - _pt_cnt.x;
        float dy = pt.y - _pt_cnt.y;
        float dd = std::hypot(dx, dy);
        if (dd > d_max) {
            continue;
        }

        float theta = std::atan2(dy, dx);
        if (theta < 0) {
            theta += PI_2;
        }

        int row = (d_max - dd) / _cell_len;
        int col = theta / reso_theta;
        if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
            continue;
        }
        _ocsc(row, col) = 1;
    }
    
    return _ocsc;
}
Eigen::MatrixXi OcSCManager::getOcSCDesc() {
    return _ocsc;
}

Eigen::MatrixXf OcSCManager::getOcSCRingKey() {
    Eigen::MatrixXf ring_key(_ocsc.rows(), 1);
    for (int i = 0; i < _ocsc.rows(); ++i) {
        ring_key(i, 0) = (float)_ocsc.row(i).sum() / (float)_cols;
    }
    return ring_key;
}

Eigen::MatrixXf OcSCManager::getOcSCSectorKey() {
    Eigen::MatrixXf sector_key(1, _ocsc.cols());
    for (int i = 0; i < _ocsc.cols(); ++i) {
        sector_key(0, i) = (float)_ocsc.col(i).sum() / (float)_rows;
    }
    return sector_key;
}

void OcSCManager::loadRawCloud(CloudPtr &cloud_in, const int nn_sam) {
    _nn_sam = nn_sam; 
    _cloud_src.reset(new PointCloud); 
    if ((int)cloud_in->points.size() > _nn_sam) {
        randomSampleCloud(cloud_in, _cloud_src, _nn_sam);
    } else {
        *_cloud_src = *cloud_in;
    }
}
void OcSCManager::loadSampleCloud(CloudPtr &cloud_in) {
    _cloud_src.reset(new PointCloud);
    *_cloud_src = *cloud_in; 
}
void OcSCManager::setPassParas(const float min_z, const float max_z) {
    _z_min = min_z;
    _z_max = max_z;
}

void OcSCManager::setCenterPt(pcl::PointXYZ &pt) {
    pcl::copyPoint(pt, _pt_cnt);
}

void OcSCManager::setCellParas(const int rows, const int cols, const float len) {
    _rows = rows;
    _cols = cols;
    _cell_len = len;
}

cv::Mat OcSCManager::getOcSCImage() {
    int rows = _ocsc.rows();
    int cols = _ocsc.cols();
    cv::Mat image(rows, cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < _ocsc.rows(); ++i) {
        for (int j = 0; j < _ocsc.cols(); ++j) {
            if (_ocsc(i, j) > 0) {
                image.at<uchar>(i, j) = 240;
            }
        }
    }
    return (image.clone());
}

cv::Mat OcSCManager::getRingKeyImage() {
    Eigen::MatrixXf ring_key(_ocsc.rows(), 1);
    for (int i = 0; i < _ocsc.rows(); ++i) {
        ring_key(i, 0) = (float)_ocsc.row(i).sum() / (float)_cols;
    }

    int rows = ring_key.rows();
    int cols = ring_key.cols();
    cv::Mat image(rows, cols, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < ring_key.rows(); ++i) {
        image.at<uchar>(i, 0) = (int)(ring_key(i, 0) * 255);
    }
    return (image.clone());
}

float distBtnOcSCsByWeightedRatioandNum(Eigen::MatrixXi& ocsc_1, 
    Eigen::MatrixXi& ocsc_2, const float& alpha, const float& beta) {
    int row = ocsc_1.rows();
    int col = ocsc_1.cols();
    int col_end = col - 1;
    std::vector <int> sims(col);
    Eigen::MatrixXi prod(row, col);
    for (int i = 0; i < col; ++i) {
        if (i == 0) {
            sims[0] = ocsc_1.cwiseProduct(ocsc_2).sum();
            continue;
        }

        Eigen::MatrixXi shift(row, col);
        shift << ocsc_1(Eigen::all, Eigen::seq(i, col_end)), ocsc_1(Eigen::all, Eigen::seq(0, (i - 1)));
        sims[i] = shift.cwiseProduct(ocsc_2).sum();
    }
    int max_value = *std::max_element(sims.begin(), sims.end());

    float loss = 1.0 - alpha * (float) max_value / (row * col) - beta * (float)max_value / (float) ocsc_1.sum();
    return loss;
}