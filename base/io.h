#ifndef _MCS_BASE_IO_H
#define _MCS_BASE_IO_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include "base/utils.hpp"
template <typename T>
bool loadMatrix44(const std::string& filename, Eigen::Matrix<T, 4, 4>& matrix) {
    matrix.setIdentity();
    std::ifstream infile(filename);
    if (!infile.is_open()) {return false;}
    infile >> matrix(0, 0) >> matrix(0, 1) >> matrix(0, 2) >> matrix(0, 3)
           >> matrix(1, 0) >> matrix(1, 1) >> matrix(1, 2) >> matrix(1, 3) 
           >> matrix(2, 0) >> matrix(2, 1) >> matrix(2, 2) >> matrix(2, 3)
           >> matrix(3, 0) >> matrix(3, 1) >> matrix(3, 2) >> matrix(3, 3);
    infile.close();
    return true;
};

template <typename T>
bool saveMatrix44(const std::string& filename, const Eigen::Matrix<T, 4, 4>& matrix) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {return false;}
    outfile << std::setprecision(15)
            << double(matrix(0, 0)) << " " << double(matrix(0, 1)) << " " << double(matrix(0, 2)) << " " << double(matrix(0, 3)) << std::endl
            << double(matrix(1, 0)) << " " << double(matrix(1, 1)) << " " << double(matrix(1, 2)) << " " << double(matrix(1, 3)) << std::endl 
            << double(matrix(2, 0)) << " " << double(matrix(2, 1)) << " " << double(matrix(2, 2)) << " " << double(matrix(2, 3)) << std::endl
            << double(matrix(3, 0)) << " " << double(matrix(3, 1)) << " " << double(matrix(3, 2)) << " " << double(matrix(3, 3));
    outfile.close();
    return true;
};
template <typename T>
bool saveDynamicMatrix(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cout << "Save dynamic matrix failed.\n";
        return false;
    }  
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            outfile << mat(i, j) << " ";
        }
        outfile << "\n";
    }
    outfile.close();
    return true;
}

template <typename T>
bool loadDynamicMatrix(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Load dynamic matrix failed.\n";
        return false;
    }  
    
    int rows = mat.rows();
    int cols = mat.cols();
    
    int count_row = 0;
    std::string str_line;
    while (std::getline(infile, str_line)) {
        std::stringstream ss(str_line);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> row_data(1, cols);
        for (int i = 0; i < cols; ++i) {
            ss >> row_data(0, i);
        }
        mat.row(count_row) = row_data;
        ++count_row;
    }
    infile.close();
    return true;
}

template <typename T>
bool saveDynamicLoopFlag(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cout << "Save dynamic loop flag failed.\n";
        return false;
    }

    int cols = mat.cols();
    if (cols > 1) {
        for (int i = 0; i < mat.rows(); ++i) {
            T sum = mat.row(i).sum();
            if (sum < 1) {
                continue;
            }
            outfile << i << " ";
            for (int j = 0; j < mat.cols(); ++j) {
                if (mat(i, j) > 0) {
                    outfile << j << " ";
                }
            }
            outfile << "\n";
        }
    } else if (cols == 1) {
        for (int i = 0; i < mat.rows(); ++i) {
            if (mat(i, 0) > 0) {
                outfile << i << "\n";
            }
        }
    } else {
        std::cout << "Dynamic loop flag cols is zero.\n";
        return false;
    }

    outfile.close();
    return true;
}

template <typename T>
bool loadDynamicLoopFlag(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Load dynamic loop flag failed.\n";
        return false;
    }  

    std::string str_line;
    while (std::getline(infile, str_line)) {
        std::stringstream ss(str_line);
        int id_row, id_col;
        ss >> id_row;
        while (ss >> id_col) {
            mat(id_row, id_col) = 1;
        }
    }
    infile.close();
    return true;
}

template <typename T>
bool saveDynamicMatrixBinary(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Save dynamic matrix failed.\n";
        return false;
    } 

    std::vector<T> vector_data(mat.data(), mat.data() + mat.size());
    outfile.write((char*)vector_data.data(), sizeof(T) * (int)vector_data.size());
    outfile.close();
    return true;
}

template <typename T>
bool loadDynamicMatrixBinary(const std::string& filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Load dynamic matrix failed.\n";
        return false;
    } 
    
    int mat_size = mat.cols() * mat.rows();
    std::vector<T> vector_data(mat_size, 0);
    infile.read((char*)vector_data.data(), sizeof(T) * mat_size);
    for (int i = 0; i < vector_data.size(); ++i) {
        int r = i / (int)mat.cols();
        int c = i % (int)mat.cols();
        mat(r, c) = vector_data[i];
    }

    infile.close();
    return true;
}
void printInfo(const int& idx, const int& step, std::string info);
#endif// _MCS_BASE_IO_H