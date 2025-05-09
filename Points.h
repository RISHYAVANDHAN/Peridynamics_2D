//
// Created by srini on 22/04/2025.
//

#ifndef POINTS_H
#define POINTS_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Points {
public:
    int Nr;
    Eigen::Vector2d X;
    Eigen::Vector2d x;
    std::vector<int> NI;
    std::vector<std::pair<int,int>> NInII;
    std::vector<Eigen::Vector2d> neighboursx;
    std::vector<Eigen::Vector2d> neighboursX;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> neighbours2x;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> neighbours2X;
    std::string Flag;
    Eigen::Vector2d BCflag;
    Eigen::Vector2d BCval;
    Eigen::Vector2d DOF;
    Eigen::Vector2d DOC;
    int n1 = 0;
    int n2 = 0;
    double volume;
    double psi;
    Eigen::Vector2d R1;
    Eigen::Vector2d R2;
    Eigen::Vector2d residual;
    Eigen::MatrixXd K1;
    Eigen::MatrixXd K2;
    Eigen::MatrixXd stiffness;
    double JI;
    double JII;

    Points();
};

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d, const std::string& DEFflag, int PD);
void neighbour_list(std::vector<Points>& point_list, double& delta);
double psifunc1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1);
double psifunc2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII, const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2);
Eigen::Vector2d PP1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1);
Eigen::Vector2d PP2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2);
Eigen::MatrixXd AA1(int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1);
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2);
void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD);
void assembly(int PD, const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag);
void update_points(int PD, std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag);

#endif // POINTS_H