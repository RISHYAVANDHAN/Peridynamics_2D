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
    int Nr;                         // Point index
    std::vector<double> X;                        // Reference coordinates
    std::vector<double> x;                        // Current coordinates
    std::vector<double> NI;                         // 1-neighbour interaction
    std::vector<std::tuple<int, int>> NInII;             // 2-neighbour interaction
    std::vector<std::vector<std::vector<int>>> neighbours;     // Neighbor list
    std::vector<std::vector<double>> neighborsx;  // Current coordinates of neighbors
    std::vector<std::vector<double>> neighborsX;  // Reference coordinates of neighbors
    std::string Flag;                // Patch/Point/Right Patch flag
    std::vector<double> BCflag{};                    // 0: Dirichlet; 1: Neumann
    std::vector<double> BCval{};                  // Boundary condition value
    std::vector<double> DOF{};                       // Global degree of freedom
    std::vector<double> DOC{};                       // Constraint flag
    int n1 = 0;                      // Number of 1-neighbour interactions
    int n2 = 0;                      // Number of 2-neighbour interactions
    double volume;                   // Volume
    double psi{};                       // Energy
    Eigen::VectorXd R1;                         // 1-neighbour residual
    Eigen::VectorXd R2;                         // 2-neighbour residual
    Eigen::VectorXd residual{};                  // Residual
    Eigen::MatrixXd K1;
    Eigen::MatrixXd K2;
    Eigen::MatrixXd stiffness{};    // Tangential stiffness per neighbor
    double JI{};                        // 1-neighbour Effective volume
    double JII{};                       // 2-neighbour effective volume

    Points();  // Default constructor
};

// Function declarations
std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d);
void neighbour_list(std::vector<Points>& point_list, double& delta);
void calculate_rk(std::vector<Points>& point_list, double C1, double delta);
void assembly(const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag);


#endif //POINTS_H
