//
// Created by srini on 22/04/2025.
//

#ifndef POINTS_H
#define POINTS_H

#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Points {
public:
    int Nr;                                     // Point index
    Eigen::Vector3d X;                        // Reference coordinates
    Eigen::Vector3d x;                        // Current coordinates
    std::vector<int> NI;                         // 1-neighbour interaction
    std::vector<std::pair<int,int>> NInII;             // 2-neighbour interaction
    //std::vector<std::vector<std::vector<int>>> neighbours;     // Neighbor list
    std::vector<Eigen::Vector3d> neighborsx;  // Current coordinates of neighbors
    std::vector<Eigen::Vector3d> neighborsX;  // Reference coordinates of neighbors
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> neighbors2x;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> neighbors2X;
    std::string Flag;                // Patch/Point/Right Patch flag
    Eigen::Vector3d BCflag{};                    // 0: Dirichlet; 1: Neumann
    Eigen::Vector3d BCval{};                  // Boundary condition value
    Eigen::Vector3d DOF{};                       // Global degree of freedom
    Eigen::Vector3d DOC{};                       // Constraint flag
    int n1 = 0;                      // Number of 1-neighbour interactions
    int n2 = 0;                      // Number of 2-neighbour interactions
    double volume;                   // Volume
    double psi{};                       // Energy
    Eigen::Vector3d R1;                         // 1-neighbour residual
    Eigen::Vector3d R2;                         // 2-neighbour residual
    Eigen::Vector3d residual{};                  // Residual
    Eigen::MatrixXd K1;
    Eigen::MatrixXd K2;
    Eigen::MatrixXd stiffness{};    // Tangential stiffness per neighbor
    double JI{};                        // 1-neighbour Effective volume
    double JII{};                       // 2-neighbour effective volume

    Points();  // Default constructor
};

// Function declarations
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d,const std::string& DEFflag, int PD);
void neighbour_list(std::vector<Points>& point_list, double& delta);
void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD);
void assembly(int PD, const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(int PD, std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag);


#endif //POINTS_H