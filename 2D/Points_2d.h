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
    int Nr;
    int NNr;  // Node number reference (index in original NL)
    int PD;
    Eigen::Vector3d X;  // Reference coordinates
    Eigen::Vector3d x;  // Current coordinates
    std::vector<int> neighbors;
    std::vector<Eigen::Vector3d> neighborsX_vec;  // Reference positions of neighbors
    std::vector<Eigen::Vector3d> neighborsx_vec;  // Current positions of neighbors
    int NI;           // Number of 1st neighbors
    int NInII;        // Number of 1st and 2nd neighbor combinations
    int NInIInIII;    // Number of 1st, 2nd and 3rd neighbor combinations
    double AV;        // Area/volume in peridynamic sense
    double Vol;       // Volume of the point
    double L;         // Lattice length
    double Delta;     // Horizon size
    int Mat;          // Material identifier
    std::vector<double> MatPars;  // Material parameters
    std::string MatLaw;           // Constitutive law
    std::vector<int> BCflg;       // Boundary condition flags
    std::vector<double> BCval;    // Boundary condition values
    std::vector<int> DOF;         // Global degree of freedom numbers
    std::string Flag;             // Point flag (e.g., "Right Patch")
    double psi;                   // Strain energy
    double residual;              // Residual force
    Eigen::VectorXd stiffness;    // Stiffness vector
    double F_ext;                 // External force

    // Constructors
    Point(int id, const Eigen::Vector3d& position) : Nr(id), X(position) {
        x = X; // Initialize current position to reference position
        NNr = 0;
        PD = 3; // Default to 3D
        NI = 0;
        NInII = 0;
        NInIInIII = 0;
        AV = 0.0;
        Vol = 0.0;
        L = 0.0;
        Delta = 0.0;
        Mat = 0;
        F_ext = 0.0;
        psi = 0.0;
        residual = 0.0;
        Flag = "";
    }
    
    Point() : Nr(0), NNr(0), PD(0), NI(0), NInII(0), NInIInIII(0), 
              AV(0.0), Vol(0.0), L(0.0), Delta(0.0), Mat(0), 
              F_ext(0.0), psi(0.0), residual(0.0), Flag("") {}
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
void calculate_rk(std::vector<Point>& PL, double C1, double delta, double nn);
void assembly(const std::vector<Point>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(std::vector<Point>& PL, double LF, Eigen::VectorXd& dx, const std::string& Update_flag, double F_prescribed, int number_of_right_patches);

#endif //POINTS_H