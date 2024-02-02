#ifndef _MCS_DESCRIPTOR_OCSC_H
#define _MCS_DESCRIPTOR_OCSC_H
#include "base/base.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class OcSCManager {
public:
    OcSCManager(){}
    ~OcSCManager(){}
    void loadRawCloud(CloudPtr &cloud_in, const int nn_sam);
    void loadSampleCloud(CloudPtr &cloud_in);

    void setCellParas(const int rows, const int cols, const float len);
    void setPassParas(const float min_z, const float max_z);
    void setCenterPt(pcl::PointXYZ &pt);
public:
    Eigen::MatrixXi makeOcSC();
    Eigen::MatrixXi getOcSCDesc(); // m*n
    Eigen::MatrixXf getOcSCRingKey(); // m*1
    Eigen::MatrixXf getOcSCSectorKey(); // 1*n

    cv::Mat getOcSCImage();
    cv::Mat getRingKeyImage();
private:
    CloudPtr _cloud_src;
    // CloudPtr _cloud_sam;
    pcl::PointXYZ _pt_cnt = pcl::PointXYZ(0.0, 0.0, 0.0);

    int _nn_sam = 8000;
    float _z_max = 2.2;
    float _z_min = -0.8;
    float _cell_len = 1.0; 
    int _rows = 20;
    int _cols = 60;
    Eigen::MatrixXi _ocsc;
};

float distBtnOcSCsByWeightedRatioandNum(Eigen::MatrixXi& ocsc_1, Eigen::MatrixXi& desc_2, 
    const float& alpha, const float& beta);

#endif //_MCS_DESCRIPTOR_OCSC_H