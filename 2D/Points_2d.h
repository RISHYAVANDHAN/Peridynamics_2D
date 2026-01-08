//
// Created by srini on 22/04/2025.
//

#ifndef POINTS_H
#define POINTS_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Point class definition to match Matlab structure and C++ implementation
class Point {
public:
    int Nr = 0;
    int NNr = 0;                 // used in Topology
    int PD = 0;

    Eigen::Vector3d X = Eigen::Vector3d::Zero();
    Eigen::Vector3d x = Eigen::Vector3d::Zero();

    std::vector<int> neighbors;
    std::vector<Eigen::Vector3d> neighborsx_vec;
    std::vector<Eigen::Vector3d> neighborsX_vec;

    int NI = 0;
    int NInII = 0;
    int NInIInIII = 0;

    double AV = 0.0;
    double Vol = 0.0;
    double L = 0.0;
    double Delta = 0.0;

    int Mat = 0;
    std::vector<double> MatPars;
    std::string MatLaw;

    std::vector<int> BCflg;      // size PD
    std::vector<double> BCval;   // size PD
    std::vector<int> DOF;        // size PD

    std::string Flag;
    double psi = 0.0;

    Eigen::VectorXd residual;
    Eigen::VectorXd stiffness;   // you write stiffness(idx)=... in calculate_rk
    Eigen::VectorXd F_ext;

    Point() = default;

    Point(int id, const Eigen::Vector3d& pos) : Nr(id), X(pos), x(pos) {}
};


// Element class definition to match Matlab structure
class Element {
public:
    int Nr;  // Element number
    std::vector<int> PNdL;  // Point node list (indices in point list)
    Eigen::MatrixXd XX;  // Node coordinates (PD x NPE matrix)
    int Mat;              // Material identifier
    std::vector<double> MatPars;  // Material parameters
    std::string MatLaw;           // Constitutive law
    
    // Constructor
    Element(int id, const std::vector<int>& nodes, const Eigen::MatrixXd& coords) : Nr(id), PNdL(nodes), XX(coords), Mat(0) {}
    Element() : Nr(0), Mat(0) {}
};

// Function declarations matching Points.cpp implementation
std::vector<Eigen::Vector3d> Compute_Corners(int PD, double SiZe);
std::pair<std::vector<Eigen::Vector3d>, std::vector<std::vector<int>>> Mesh(int PD, const std::vector<Eigen::Vector3d>& Corners, double L);
bool PatchNode(const Eigen::VectorXd& node, const std::vector<Eigen::Vector3d>& Corners);
std::vector<Eigen::Vector3d> Patch(int PD, const std::vector<Eigen::Vector3d>& Corners, double L, double Delta, int patch, int right_patch);
std::pair<std::vector<Point>, std::vector<Element>> Topology(int PD, const std::vector<Eigen::Vector3d>& NL, const std::vector<std::vector<int>>& CNCT, double L, double Delta, const std::vector<double>& Bvals, const std::string& TOPflag);
std::vector<Point> AssignNgbrs(int PD, std::vector<Point> PL, double L, double Delta);
std::vector<Point> AssignVols(const std::vector<Eigen::Vector3d>& Corners, std::vector<Point> PL, double L, const std::vector<double>& Bvals, const std::string& TOPflag);

// SetMaterial overloaded functions
std::vector<Point> SetMaterial(const std::vector<Point>& inp, double L, double Delta, const std::vector<double>& MatPars, const std::string& MatLaw);
std::vector<Element> SetMaterial(const std::vector<Element>& inp, const std::vector<double>& MatPars, const std::string& MatLaw);

Eigen::Matrix3d Compute_FF(int PD, double d, const std::string& DEFflag);
std::vector<Point> FreeAllPoints(std::vector<Point> PL);
std::pair<std::vector<Point>, int> AssignGlobalDOF(std::vector<Point> PL);
std::pair<std::vector<Point>, int> AssignBCs(const std::vector<Eigen::Vector3d>& Corners, std::vector<Point> PL, const Eigen::Matrix3d& FF, const std::string& BCflag, const std::string& PatchFlag);
void calculate_rk(std::vector<Point>& PL, double C1, double C2, double delta, int PD);
void assembly(const std::vector<Point>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(std::vector<Point>& PL, double LF, Eigen::VectorXd& dx, const std::string& Update_flag, double F_prescribed, int number_of_right_patches);
void debug_point_29(const std::vector<Point>& PL, int point_idx, double C1, double C2, double Delta, int PD);
#endif //POINTS_H