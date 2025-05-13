#include "Points.h"
#include <iostream>
#include <cmath>
#include <set>
#include <Eigen/Sparse>

Points::Points()
    : Nr(0), X(Eigen::Vector2d::Zero()), x(Eigen::Vector2d::Zero()),
      NI(), NInII(), neighboursx(), neighboursX(), neighbours2x(),
      neighbours2X(), Flag(""), BCflag(Eigen::Vector2i::Zero()),
      BCval(Eigen::Vector2d::Zero()), DOF(Eigen::Vector2i::Zero()),
      DOC(Eigen::Vector2i::Zero()), n1(0), n2(0), volume(0.0), psi(0.0),
      R1(Eigen::Vector2d::Zero()), R2(Eigen::Vector2d::Zero()),
      residual(Eigen::Vector2d::Zero()), K1(), K2(), stiffness(),
      JI(0.0), JII(0.0) {}

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d, const std::string& DEFflag, int PD) 
{
    std::vector<Points> point_list;
    const int number_of_points_x = std::floor(domain_size / Delta) + 1;
    const int number_of_points_y = number_of_points_x; // Adjust if non-square needed
    const int total_points_x = number_of_patches + number_of_points_x + number_of_right_patches;
    int index = 0;

    Eigen::Matrix2d FF = Eigen::Matrix2d::Identity();
    if (DEFflag == "EXT") FF(0, 0) = 1 + d;
    else if (DEFflag == "EXP") FF = (1 + d) * Eigen::Matrix2d::Identity();
    else if (DEFflag == "SHR") FF(1, 0) = d;

    std::cout<<FF<<std::endl;

    for (int j = 0; j < number_of_points_y; j++) {
        for (int i = 0; i < total_points_x; i++) {
            Points point;
            point.Nr = index++;
            point.X = {(Delta / 2) + i * Delta, (Delta / 2) + j * Delta};
            point.x = point.X;

            if (point.X(0) < (number_of_patches * Delta)) {
                point.Flag = "Patch";
                point.BCval = Eigen::Vector2d::Zero();
                point.BCflag.setOnes();
                point.DOC << ++DOCs, ++DOCs;
            } else if (point.X(0) >= Delta * (number_of_patches + number_of_points_x)) {
                point.Flag = "RightPatch";
                point.BCval << d, d; 
                point.BCflag.setOnes();
                point.DOC << ++DOCs, ++DOCs;
            } else {
                point.Flag = "Point";
                point.BCval = (FF * point.X) - point.X;
                point.BCflag << 0,0;
                point.DOF << ++DOFs, ++DOFs;
            }

            point.volume = 1.0;
            point_list.push_back(point);
        }
    }
    return point_list;
}

void neighbour_list(std::vector<Points>& point_list, double delta) 
{
    for (auto& i : point_list) {
        i.NI.clear(); i.NInII.clear();
        i.neighboursx.clear(); i.neighboursX.clear();
        i.neighbours2x.clear(); i.neighbours2X.clear();
        i.n1 = i.n2 = 0;
        std::set<std::pair<int, int>> unique_pairs;

        // Find 1-neighbors
        for (const auto& j : point_list) {
            if ((i.Nr != j.Nr) && ((i.X - j.X).norm() < delta)) {
                i.NI.push_back(j.Nr);
                i.neighboursx.push_back(j.x);
                i.neighboursX.push_back(j.X);
                i.n1++;
            }
        }

        // Find 2-neighbor pairs
        for (size_t p = 0; p < i.NI.size(); ++p) {
            for (size_t q = p+1; q < i.NI.size(); ++q) { // Prevent duplicate checks
                Eigen::Vector2d XiI = point_list[i.NI[p]].X - i.X;
                Eigen::Vector2d XiII = point_list[i.NI[q]].X - i.X;
                
                double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
                Eigen::Vector2d l_vec = XiI - XiII;
                
                if (std::abs(A) < 1e-6 && l_vec.norm() < delta) {
                    auto sorted_pair = std::minmax(i.NI[p], i.NI[q]);
                    if (unique_pairs.insert(sorted_pair).second) {
                        i.NInII.push_back(sorted_pair);
                        i.neighbours2X.emplace_back(point_list[sorted_pair.first].X, point_list[sorted_pair.second].X);
                        i.neighbours2x.emplace_back(point_list[sorted_pair.first].x, point_list[sorted_pair.second].x);
                        i.n2++;
                    }
                }
            }
        }
        //std::cout << "Point " << i.Nr << ": 1-neighbors = " << i.n1 << ", 2-neighbors = " << i.n2 << std::endl;
    }
}

double psifunc1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    if (L < 1e-12) return 0.0; // Prevent division by zero
    double l = xiI.norm();
    double s = (l - L)/L;
    return C1 * 0.5 * s * s;
}

Eigen::Vector2d PP1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    if (L < 1e-12 || xiI.norm() < 1e-12) return Eigen::Vector2d::Zero();
    Eigen::Vector2d eta = xiI.normalized();
    return C1 * eta * ((xiI.norm() - L)/L);
}

double psifunc2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII, const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, double C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    if (std::abs(A) < 1e-12) return 0.0;
    return C2 * (1.0/3.0) * std::pow((std::abs(A)-std::abs(a))/std::abs(A), 2) * std::abs(A);
}

Eigen::Vector2d PP2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII, const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, double C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    if (std::abs(A) < 1e-12 || std::abs(a) < 1e-12) return Eigen::Vector2d::Zero();
    double G = (1/std::abs(A)) - (1/std::abs(a));
    Eigen::Vector2d H = (xiII.squaredNorm() * xiI) - (xiII.dot(xiI) * xiII);
    return 2.0 * C2 * G * H;
}

Eigen::MatrixXd AA1(int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, double C1) {
    double L = XiI.norm();
    double l = xiI.norm();
    if (L < 1e-12 || l < 1e-12) return Eigen::MatrixXd::Zero(PD, PD);
    Eigen::Vector2d eta = xiI.normalized();
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd eta_outer = eta * eta.transpose();
    return C1 * (( (l - L) / (L * l) ) * (II - eta_outer) + (1.0 / L) * eta_outer);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII, const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, double C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    if (std::abs(A) < 1e-12 || std::abs(a) < 1e-12) 
        return {Eigen::MatrixXd::Zero(PD, PD), Eigen::MatrixXd::Zero(PD, PD)};
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd BBI1 = (xiII.dot(xiII) * II) - (xiII * xiII.transpose());
    Eigen::MatrixXd BBJ1 = (2 * xiI * xiII.transpose()) - (xiI.dot(xiII) * II) - (xiII * xiI.transpose());

    Eigen::Vector2d eInII = (xiII.squaredNorm() * xiI) - (xiI.dot(xiII) * xiII);
    Eigen::Vector2d eIInI = (xiI.squaredNorm() * xiII) - (xiII.dot(xiI) * xiI);

    Eigen::MatrixXd AA2I = (2*C2*((1/std::abs(A) - 1/std::abs(a))) * BBI1 + 
                          (2*C2/std::pow(std::abs(a),3)) * (eInII * eInII.transpose()));
    Eigen::MatrixXd AA2J = (2*C2*((1/std::abs(A) - 1/std::abs(a))) * BBJ1 +
                          (2*C2/std::pow(std::abs(a),3)) * (eInII * eIInI.transpose()));

    return {AA2I, AA2J};
}

void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD)
{
    constexpr double pi = 3.14159265358979323846;
    const double Vh = pi * std::pow(delta, 2); // Horizon volume for 2D

    for (auto& pt : point_list)
    {
        // Reset all force and stiffness values
        pt.residual.setZero();
        pt.R1.setZero();
        pt.R2.setZero();
        pt.psi = 0.0;
        pt.K1.setZero();
        pt.K2.setZero();
        pt.stiffness.setZero();

        // Calculate weighting factors
        const double JI = (pt.n1 > 0) ? Vh / pt.n1 : 0.0;
        const double JInII = (pt.n2 > 0) ? (Vh * Vh) / pt.n2 : 0.0;

        // Create extended neighbor list (1-neighbors + self)
        std::vector<int> NNgbrEI = pt.NI;
        NNgbrEI.push_back(pt.Nr);
        const int num_NNgbrEI = NNgbrEI.size();

        // Create extended pair list (2-neighbors + self pair)
        std::vector<std::pair<int, int>> NNgbrEII = pt.NInII;
        NNgbrEII.emplace_back(pt.Nr, pt.Nr);
        const int num_NNgbrEII = NNgbrEII.size();

        // Initialize stiffness matrix with proper dimensions
        const int total_cols = num_NNgbrEI + 2 * num_NNgbrEII;
        pt.stiffness = Eigen::MatrixXd::Zero(PD*PD, total_cols);
        pt.K1 = Eigen::MatrixXd::Zero(PD*PD, total_cols);
        pt.K2 = Eigen::MatrixXd::Zero(PD*PD, total_cols);

        // ================== 1-NEIGHBOR INTERACTIONS ==================
        for (size_t j = 0; j < pt.n1; j++)
        {
            // Get neighbor reference and current configuration vectors
            const Eigen::Vector2d XiI = pt.neighboursX[j] - pt.X;
            const Eigen::Vector2d xiI = pt.neighboursx[j] - pt.x;

            // Calculate bond stretch energy and force
            const double psi1 = psifunc1(XiI, xiI, C1);
            const Eigen::Vector2d R1_temp = PP1(XiI, xiI, C1);
            
            // Accumulate energy and force
            pt.psi += JI * psi1;
            pt.R1 += JI * R1_temp;
            //std::cout << "R1_temp: " << R1_temp.transpose() << std::endl;   

            // Calculate stiffness contribution for each extended neighbor
            for (int b = 0; b < num_NNgbrEI; b++)
            {
                // Determine if current neighbor contributes to this column
                double K_factor = 0.0;
                if (pt.NI[j] == NNgbrEI[b]) K_factor += 1.0;
                if (pt.Nr == NNgbrEI[b]) K_factor -= 1.0;

                // Compute stiffness block only for significant contributions
                if (std::abs(K_factor) > 1e-12)
                {
                    const Eigen::MatrixXd AA1_block = AA1(PD, XiI, xiI, C1);
                    const Eigen::MatrixXd stiffness_block = AA1_block * JI * K_factor;

                    // Map matrix to vector column-wise
                    pt.K1.block(0, b, PD*PD, 1) = Eigen::Map<const Eigen::VectorXd>(stiffness_block.data(), PD*PD);
                }
            }
        }

        // ================== 2-NEIGHBOR INTERACTIONS ==================
        for (size_t j = 0; j < pt.NInII.size(); j++)
        {
            // Get pair information from reference configuration
            const int pair_first = pt.NInII[j].first;
            const int pair_second = pt.NInII[j].second;
            
            // Get current configuration vectors
            const Eigen::Vector2d XiI  = pt.neighbours2X[j].first  - pt.X;
            const Eigen::Vector2d XiII = pt.neighbours2X[j].second - pt.X;
            const Eigen::Vector2d xiI  = pt.neighbours2x[j].first  - pt.x;
            const Eigen::Vector2d xiII = pt.neighbours2x[j].second - pt.x;

            // Calculate pair energy and force
            const double psi2 = psifunc2(XiI, XiII, xiI, xiII, C2);
            const Eigen::Vector2d R2_temp = PP2(XiI, XiII, xiI, xiII, C2);

            // Accumulate energy and force
            pt.psi += JInII * psi2;
            pt.R2 += JInII * R2_temp;
            //std::cout << "R2_temp: " << R2_temp.transpose() << std::endl;

            // Calculate stiffness contributions for each extended pair
            for (int b = 0; b < num_NNgbrEII; b++)
            {
                // First member contribution
                double K_factor_i = 0.0;
                if (pt.NInII[j].first == NNgbrEII[b].first) 
                    K_factor_i += 1.0;
                if (pt.Nr == NNgbrEII[b].first) 
                    K_factor_i -= 1.0;

                // Second member contribution
                double K_factor_j = 0.0;
                if (pt.NInII[j].second == NNgbrEII[b].second) 
                    K_factor_j += 1.0;
                if (pt.Nr == NNgbrEII[b].second) 
                    K_factor_j -= 1.0;

                // Only compute non-zero contributions
                if (std::abs(K_factor_i) > 1e-12 || std::abs(K_factor_j) > 1e-12)
                {
                    auto [AA2I, AA2J] = AA2(PD, XiI, XiII, xiI, xiII, C2);
                    
                    // Map matrices to vector columns
                    const int col_idx = num_NNgbrEI + 2*b;
                    if (col_idx < total_cols)
                    {
                        pt.K2.block(0, col_idx, PD*PD, 1) = Eigen::Map<Eigen::VectorXd>(AA2I.data(), PD*PD) * JInII * K_factor_i;
                    }
                    
                    if (col_idx+1 < total_cols)
                    {
                        pt.K2.block(0, col_idx+1, PD*PD, 1) = Eigen::Map<Eigen::VectorXd>(AA2J.data(), PD*PD) * JInII * K_factor_j;
                    }
                }
            }
        }

        // ================== COMBINE CONTRIBUTIONS ==================
        pt.residual = pt.R1 + pt.R2;
        pt.stiffness = pt.K1 + pt.K2;

        // Numerical stability check
        if (!pt.residual.allFinite() || !pt.stiffness.allFinite()) {
            std::cerr << "NaN/Inf detected in point " << pt.Nr << std::endl;
            throw std::runtime_error("Numerical instability in calculate_rk");
        }
        //std::cout << "Point " << pt.Nr << ": Residual = "<< std::endl << pt.residual.transpose() << std::endl; //<< ", Stiffness = " << std::endl<< pt.stiffness<< std::endl;
    }
}

void assembly(const int PD, const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag) 
{
    if (flag == "residual") {
        R = Eigen::VectorXd::Zero(DOFs);
        for (const auto& point : point_list) {
            for (int d = 0; d < PD; d++) {
                if (point.BCflag(d) == 0 && point.DOF(d) > 0 && point.DOF(d) <= DOFs) {
                    const int index = point.DOF(d) - 1;
                    R(index) += point.residual(d);
                }
            }
        }
    }
    else if (flag == "stiffness") {
        std::vector<Eigen::Triplet<double>> triplets;
        //triplets.reserve(point_list.size() * 4 * PD * PD); // Preallocate memory

        for (size_t pt_idx = 0; pt_idx < point_list.size(); pt_idx++) {
            const auto& pt = point_list[pt_idx];
            if (pt.stiffness.cols() == 0) continue;

            // Extended neighbor list: 1-neighbors + self
            std::vector<int> NNgbrEI = pt.NI;
            NNgbrEI.push_back(pt.Nr);

            // Process 1-neighbor interactions
            for (size_t col = 0; col < NNgbrEI.size(); col++) {
                const int q = NNgbrEI[col];
                if (q < 0 || q >= point_list.size()) continue;

                // Extract stiffness block for this column
                Eigen::Map<const Eigen::MatrixXd> K_block(pt.stiffness.col(col).data(), PD, PD);

                // Add contributions to global matrix
                for (int i = 0; i < PD; i++) {
                    if (pt.BCflag(i) == 0 || pt.DOF(i) >= 0) continue;
                    const int row = pt.DOF(i) - 1;

                    const auto& nbr_pt = point_list[q];
                    for (int j = 0; j < PD; j++) {
                        if (nbr_pt.BCflag(j) == 0 && nbr_pt.DOF(j) > 0) {
                            const int col_idx = nbr_pt.DOF(j) - 1;
                            if (std::isfinite(K_block(i,j))) {
                                triplets.emplace_back(row, col_idx, K_block(i,j));
                            }
                        }
                    }
                }
            }
            std::vector<std::pair<int, int>> NNgbrEII = pt.NInII;
            NNgbrEII.emplace_back(pt.Nr, pt.Nr);
            const int num_NNgbrEII = NNgbrEII.size();

            // Process 2-neighbor pair interactions
            for (size_t pair_idx = 0; pair_idx < num_NNgbrEII; pair_idx++) {
                const auto& pair = NNgbrEII[pair_idx];
                const int col_q = NNgbrEI.size() + 2 * pair_idx;
                const int col_r = col_q + 1;

                // First member of pair
                if (col_q < pt.stiffness.cols()) {
                    Eigen::Map<const Eigen::MatrixXd> K_block_q(pt.stiffness.col(col_q).data(), PD, PD);
                    const int q = pair.first;
                    if (q >= 0 && q < point_list.size()) {
                        const auto& nbr_q = point_list[q];
                        for (int i = 0; i < PD; i++) {
                            if (pt.BCflag(i) == 0 || pt.DOF(i) >= 0) continue;
                            const int row = pt.DOF(i) - 1;
                            for (int j = 0; j < PD; j++) {
                                if (nbr_q.BCflag(j) == 0 && nbr_q.DOF(j) > 0) {
                                    const int col = nbr_q.DOF(j) - 1;
                                    if (std::isfinite(K_block_q(i,j))) {
                                        triplets.emplace_back(row, col, K_block_q(i,j));
                                    }
                                }
                            }
                        }
                    }
                }

                // Second member of pair
                if (col_r < pt.stiffness.cols()) {
                    Eigen::Map<const Eigen::MatrixXd> K_block_r(pt.stiffness.col(col_r).data(), PD, PD);
                    const int r = pair.second;
                    if (r >= 0 && r < point_list.size()) {
                        const auto& nbr_r = point_list[r];
                        for (int i = 0; i < PD; i++) {
                            if (pt.BCflag(i) == 0 || pt.DOF(i) >= 0) continue;
                            const int row = pt.DOF(i) - 1;
                            for (int j = 0; j < PD; j++) {
                                if (nbr_r.BCflag(j) == 0 && nbr_r.DOF(j) > 0) {
                                    const int col = nbr_r.DOF(j) - 1;
                                    if (std::isfinite(K_block_r(i,j))) {
                                        triplets.emplace_back(row, col, K_block_r(i,j));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());
        K.makeCompressed();
    }
}

void update_points(const int PD, std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag, double delta, int DOFs)
{
    if (Update_flag == "Prescribed") {
        for (auto& i : point_list) {
            for (int p = 0; p < PD; p++) {
                if (i.BCflag(p) == 1) {
                    i.x(p) = i.X(p) + (LF * i.BCval(p));
                    //std::cout << "Updated prescribed point " << i.Nr << " coord " << p << " to " << i.x[p] << std::endl;
                }
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (auto& i : point_list) {
            for (int q = 0; q < PD; q++) {
                if (i.BCflag(q) == 0 && i.DOF(q) > 0 && i.DOF(q) <= DOFs) {
                    i.x(q) += dx(i.DOF(q) - 1);
                    //std::cout << "Updated point " << i.Nr << " coord " << q << " by " << dx[i.DOF[q] - 1] << std::endl;
                }
            }
        }
    }

    // Update neighbor coordinates
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
            if (idx1 >= 0 && idx1 < point_list.size() && idx2 >= 0 && idx2 < point_list.size()) {
                point.neighbours2x[i] = std::make_pair( point_list[idx1].x, point_list[idx2].x);
            }
        }
    }
    neighbour_list(point_list, delta);
}