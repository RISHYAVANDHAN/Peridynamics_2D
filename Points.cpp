#include "Points.h"
#include <iostream>
#include <cmath>
#include <Eigen/Sparse>

Points::Points():
    Nr(0),
    X(Eigen::Vector2d::Zero()), x(Eigen::Vector2d::Zero()),
      BCflag(Eigen::Vector2d::Zero()), BCval(Eigen::Vector2d::Zero()),
      DOF(Eigen::Vector2d::Zero()), DOC(Eigen::Vector2d::Zero()),
      R1(Eigen::Vector2d::Zero()), R2(Eigen::Vector2d::Zero()),
      residual(Eigen::Vector2d::Zero()), volume(0.0), psi(0.0),
      JI(0.0), JII(0.0) {}

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta,
                        int number_of_right_patches, int& DOFs, int& DOCs,
                        double d, const std::string& DEFflag, int PD) {
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;

    Eigen::Matrix2d FF = Eigen::Matrix2d::Identity();

    if (DEFflag == "EXT") {
        FF(0, 0) = 1 + d;
    }
    else if (DEFflag == "EXP") {
        FF = (1 + d) * Eigen::Matrix2d::Identity();
    }
    else if (DEFflag == "SHR") {
        FF(1, 0) = d;
    }

    for (int j = 0; j < number_of_points; j++) {
        for (int i = 0; i < total_points; i++) {
            Points point;
            point.Nr = index++;
            point.X = Eigen::Vector2d((Delta/2) + i*Delta, (Delta/2) + j*Delta);
            point.x = point.X;

            if (point.X[0] < (number_of_patches * Delta)) {
                point.Flag = "Patch";
                point.BCval = Eigen::Vector2d::Zero();
                point.BCflag = Eigen::Vector2d::Zero();
                point.DOC[0] = ++DOCs;
            }
            else if (point.X[0] > (Delta * (number_of_points + number_of_patches))) {
                point.Flag = "RightPatch";
                point.BCval = Eigen::Vector2d(d, 0);
                point.BCflag = Eigen::Vector2d::Zero();
                point.DOC[0] = ++DOCs;
            }
            else {
                point.Flag = "Point";
                point.BCflag = Eigen::Vector2d(1, 0);
                point.BCval = (FF * point.X) - point.X;
                point.DOF[0] = ++DOFs;
            }

            point.volume = 1;
            point_list.push_back(point);
        }
    }

    DOFs = 0;
    for (auto& point : point_list) {
        if (point.BCflag[0] == 1) {
            point.DOF[0] = ++DOFs;
        }
    }

    return point_list;
}

void neighbour_list(std::vector<Points>& point_list, double& delta) {
    for (auto &i : point_list) {
        i.neighboursx.clear();
        i.neighboursX.clear();
        i.neighbours2x.clear();
        i.neighbours2X.clear();
        i.NI.clear();
        i.NInII.clear();
        i.n1 = i.n2 = 0;

        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) && ((i.X - j.X).norm() < delta)) {
                i.NI.push_back(j.Nr);
                i.neighboursx.push_back(j.x);
                i.neighboursX.push_back(j.X);
                i.n1++;
            }
        }

        for (size_t p = 0; p < i.NI.size(); ++p) {
            for (size_t q = 0; q < i.NI.size(); ++q) {
                if (i.NI[p] != i.NI[q]) {
                    Eigen::Vector2d XiI = i.neighboursX[p] - i.X;
                    Eigen::Vector2d XiII = i.neighboursX[q] - i.X;
                    Eigen::Vector2d l = XiI - XiII;

                    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
                    if ((std::abs(A) < 1e-6) && l.norm() < delta) {
                        i.NInII.push_back(std::make_pair(i.NI[p], i.NI[q]));
                        i.neighbours2x.push_back(std::make_pair(point_list[i.NI[p]].x,
                                                point_list[i.NI[q]].x));
                        i.neighbours2X.push_back(std::make_pair(point_list[i.NI[p]].X,
                                                point_list[i.NI[q]].X));
                        i.n2++;
                    }
                }
            }
        }
    }
}

double psifunc1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    double l = xiI.norm();
    return C1 * 0.5 * std::pow((l - L)/L, 2);
}

double psifunc2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,
               const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    return C2 * (1.0/3.0) * std::pow((std::abs(A) - std::abs(a))/std::abs(A), 2);
}

Eigen::Vector2d PP1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    double l = xiI.norm();
    Eigen::Vector2d eta = xiI.normalized();
    return C1 * ((l - L)/L) * eta;
}

Eigen::Vector2d PP2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,
                  const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    double G = (1/std::abs(A)) - (1/std::abs(a));

    Eigen::Vector2d H = xiII.squaredNorm() * xiI - xiII.dot(xiI) * xiII;
    return 2.0 * C2 * G * H;
}

Eigen::MatrixXd AA1(int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    double l = xiI.norm();
    Eigen::Vector2d eta = xiI.normalized();
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd eta_outer = eta * eta.transpose();

    return C1 * (( (l-L)/(L*l) ) * (II - eta_outer) + (1.0/L) * eta_outer);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(int PD, const Eigen::Vector2d& XiI,
                                              const Eigen::Vector2d& XiII,
                                              const Eigen::Vector2d& xiI,
                                              const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    double AA = std::abs(A);
    double aa = std::abs(a);

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd BBI1 = (xiII.dot(xiII) * II) - (xiII * xiII.transpose());
    Eigen::MatrixXd BBJ1 = (2 * xiI * xiII.transpose()) -
                          (xiI.dot(xiII) * II) -
                          (xiII * xiI.transpose());

    Eigen::Vector2d eInII = (xiII.dot(xiII) * xiI - (xiI.dot(xiII) * xiII));
    Eigen::Vector2d eIInI = (xiI.dot(xiI) * xiII - (xiII.dot(xiI) * xiI));

    Eigen::MatrixXd AA2I = (2 * C2 * ((1/AA) - (1/aa)) * BBI1) +
                          (2 * C2 * (1/std::pow(aa,3)) * (eInII * eInII.transpose()));

    Eigen::MatrixXd AA2J = (2 * C2 * ((1/AA) - (1/aa)) * BBJ1) +
                          (2 * C2 * (1/std::pow(aa,3)) * (eInII * eIInI.transpose()));

    return {AA2I, AA2J};
}

void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD) {
    constexpr double pi = 3.14159265358979323846;
    double Vh = pi * std::pow(delta, 2);

    for (auto& i : point_list) {
        i.residual = Eigen::Vector2d::Zero();
        i.R1 = Eigen::Vector2d::Zero();
        i.R2 = Eigen::Vector2d::Zero();
        i.psi = 0;

        double JI = Vh / i.n1;
        double JInII = (Vh * Vh) / i.n2;

        std::vector<int> neighboursEI = i.NI;
        std::vector<std::pair<int, int>> neighboursEII = i.NInII;
        std::vector<Eigen::Vector2d> neighboursEx = i.neighboursx;
        std::vector<Eigen::Vector2d> neighboursEX = i.neighboursX;

        neighboursEI.push_back(i.Nr);
        neighboursEII.push_back(std::make_pair(i.Nr,i.Nr));
        neighboursEx.push_back(i.x);
        neighboursEX.push_back(i.X);

        const int NNgbrEI = neighboursEI.size();
        const int NNgbrEII = neighboursEII.size();
        int total_cols = NNgbrEI + 2 * NNgbrEII;

        i.stiffness = Eigen::MatrixXd::Zero(PD * PD, total_cols);
        i.K1 = Eigen::MatrixXd::Zero(PD * PD, total_cols);
        i.K2 = Eigen::MatrixXd::Zero(PD * PD, total_cols);

        for (size_t j = 0; j < i.n1; j++) {
            Eigen::Vector2d XiI = i.neighboursX[j] - i.X;
            Eigen::Vector2d xiI = i.neighboursx[j] - i.x;

            double psi1 = psifunc1(XiI, xiI, C1);
            Eigen::Vector2d R1_temp = PP1(XiI, xiI, C1);
            i.psi += JI * psi1;
            i.residual += JI * R1_temp;

            for (int b = 0; b < NNgbrEI; b++) {
                double K_factor = 0.0;
                if (i.NI[j] == neighboursEI[b]) K_factor += 1.0;
                if (i.Nr == neighboursEI[b]) K_factor -= 1.0;

                Eigen::MatrixXd AA1I = AA1(PD, XiI, xiI, C1);
                Eigen::MatrixXd stiffness_contribution = AA1I * JI * K_factor;
                i.K1.block(0, b, PD * PD, 1) = Eigen::Map<Eigen::VectorXd>(stiffness_contribution.data(), PD * PD);
            }
        }

        for (size_t j = 0; j < i.NInII.size(); j++) {
            Eigen::Vector2d XiI = (i.neighbours2X[j]).first - i.X;
            Eigen::Vector2d xiI = (i.neighbours2x[j]).first - i.x;
            Eigen::Vector2d XiII = (i.neighbours2X[j]).second - i.X;
            Eigen::Vector2d xiII = (i.neighbours2x[j]).second - i.x;

            double psi2 = psifunc2(XiI, XiII, xiI, xiII, C2);
            Eigen::Vector2d R2_temp = PP2(XiI, XiII, xiI, xiII, C2);
            i.psi += JInII * psi2;
            i.residual += JInII * R2_temp;

            for (int b = 0; b < NNgbrEII; b++) {
                double K_factor_i = 0.0;
                if ((i.NInII[j]).first == (neighboursEII[b]).first) K_factor_i += 1.0;
                if (i.Nr == (neighboursEII[b]).first) K_factor_i -= 1.0;

                double K_factor_j = 0.0;
                if ((i.NInII[j]).second == (neighboursEII[b]).second) K_factor_j += 1.0;
                if (i.Nr == (neighboursEII[b]).second) K_factor_j -= 1.0;

                auto [AA2I, AA2J] = AA2(PD, XiI, XiII, xiI, xiII, C2);
                Eigen::MatrixXd stiff_i = AA2I * K_factor_i * JInII;
                Eigen::MatrixXd stiff_j = AA2J * K_factor_j * JInII;

                i.K2.block(0, NNgbrEI + 2 * b, PD * PD, 1) = Eigen::Map<Eigen::VectorXd>(stiff_i.data(), PD * PD);
                i.K2.block(0, NNgbrEI + 2 * b + 1, PD * PD, 1) = Eigen::Map<Eigen::VectorXd>(stiff_j.data(), PD * PD);
            }
        }

        i.residual = i.R1 + i.R2;
        i.stiffness = i.K1 + i.K2;
    }
}

void assembly(int PD, const std::vector<Points>& point_list, int DOFs,
             Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag) {
    if (flag == "residual") {
        R.setZero();
        for (const auto& point : point_list) {
            for (int ii = 0; ii < PD; ii++) {
                if (point.BCflag[ii] == 1 && point.DOF[ii] > 0) {
                    R[point.DOF[ii] - 1] += point.residual[ii];
                }
            }
        }
    }
    else if (flag == "stiffness") {
        std::vector<Eigen::Triplet<double>> triplets;
        K.setZero();

        for (const auto& point : point_list) {
            for (int pp = 0; pp < PD; pp++) {
                if (point.BCflag[pp] == 1 && point.DOF[pp] > 0) {
                    std::vector<int> neighboursEI = point.NI;
                    neighboursEI.push_back(point.Nr);

                    for (size_t q = 0; q < neighboursEI.size(); q++) {
                        int nbr_idx = neighboursEI[q];
                        if (nbr_idx < 0 || nbr_idx >= point_list.size()) continue;

                        Eigen::MatrixXd K_PQ = Eigen::Map<const Eigen::MatrixXd>(
                            point.stiffness.col(q).data(), PD, PD);

                        for (int qq = 0; qq < PD; qq++) {
                            if (point_list[nbr_idx].BCflag[qq] == 1 &&
                                point_list[nbr_idx].DOF[qq] > 0) {
                                triplets.emplace_back(
                                    point.DOF[pp] - 1,
                                    point_list[nbr_idx].DOF[qq] - 1,
                                    K_PQ(pp, qq)
                                );
                            }
                        }
                    }
                }
            }
        }
        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());
    }
}

void update_points(int PD, std::vector<Points>& point_list, double LF,
                  Eigen::VectorXd& dx, const std::string& Update_flag) {
    if (Update_flag == "Prescribed") {
        for (auto& i : point_list) {
            for (int p = 0; p < PD; p++) {
                if (i.BCflag[p] == 0) {
                    i.x += LF * i.BCval;
                }
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (auto& i : point_list) {
            for (int q = 0; q < PD; q++) {
                if (i.BCflag[q] == 1 && i.DOF[q] > 0 && i.DOF[q] <= dx.size()) {
                    i.x[q] += dx[i.DOF[q] - 1];
                }
            }
        }
    }

    for (auto& point : point_list) {
        for (size_t i = 0; i < point.NI.size(); i++) {
            int nbr_idx = point.NI[i];
            if (nbr_idx >= 0 && nbr_idx < point_list.size()) {
                point.neighboursx[i] = point_list[nbr_idx].x;
            }
        }

        for (size_t i = 0; i < point.NInII.size(); i++) {
            int idx1 = point.NInII[i].first;
            int idx2 = point.NInII[i].second;
            if (idx1 >= 0 && idx1 < point_list.size() &&
                idx2 >= 0 && idx2 < point_list.size()) {
                point.neighbours2x[i] = {point_list[idx1].x, point_list[idx2].x};
            }
        }
    }
}