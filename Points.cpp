
#include "Points.h"
#include <iostream>
#include <cmath>
#include <Eigen/Sparse>

// Constructor with same initialization pattern
Points::Points()
    : Nr(0),
      X(Eigen::Vector2d::Zero()),      // 2D zero vector
      x(Eigen::Vector2d::Zero()),
      NI(),
      NInII(),
      neighboursx(),
      neighboursX(),
      neighbours2x(),
      neighbours2X(),
      Flag(""),
      BCflag(Eigen::Vector2d::Zero()), // 2D BC flags
      BCval(Eigen::Vector2d::Zero()),
      DOF(Eigen::Vector2d::Zero()),
      DOC(Eigen::Vector2d::Zero()),
      n1(0),
      n2(0),
      volume(0.0),
      psi(0.0),
      R1(Eigen::Vector2d::Zero()),     // 2D residuals
      R2(Eigen::Vector2d::Zero()),
      residual(Eigen::Vector2d::Zero()),
      K1(),
      K2(),
      stiffness(),
      JI(0.0),
      JII(0.0)
{}

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d, const std::string& DEFflag, int PD)
{
    // Original mesh generation logic with 2D coordinates
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;

    Eigen::Matrix2d FF = Eigen::Matrix2d::Identity(); // 2x2 deformation gradient

    if (DEFflag == "EXT") {
        FF(0, 0) = 1 + d;
    }
    else if (DEFflag == "EXP") {
        FF = (1 + d) * Eigen::Matrix2d::Identity();
    }
    else if (DEFflag == "SHR") {
        FF(1, 0) = d;
    }

    for (int j = 0; j < number_of_points; j++)
    {
        for (int i = 0; i < total_points; i++)
        {
            Points point;
            point.Nr = index++;
            // 2D coordinates without Z-component
            point.X = {(Delta / 2) + i * Delta, (Delta / 2) + j * Delta};
            point.x = point.X;

            if (point.X[0] < (number_of_patches * Delta))
            {
                point.Flag = "Patch";
                point.BCval = Eigen::Vector2d::Zero();
                point.BCflag = Eigen::Vector2d::Zero();
                point.DOC[0] = ++DOCs;
                //point.DOC[1] = ++DOCs;

            }
            else if ((point.X[0] > (Delta * (number_of_points + number_of_patches))))
            {
                point.Flag = "RightPatch";
                point.BCval = Eigen::Vector2d(d, 0);
                point.BCflag = Eigen::Vector2d::Zero();
                point.DOC[0] = ++DOCs;
                //point.DOC[1] = ++DOCs;
            }
            else
            {
                point.Flag = "Point";
                point.BCflag = Eigen::Vector2d(1,0);
                point.BCval = (FF * point.X) - point.X;
                point.DOF[0] = ++DOFs;
                //point.DOF[1] = ++DOFs;
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

void neighbour_list(std::vector<Points>& point_list, double& delta)
{
    // Original neighbor list logic with 2D checks
    for (auto &i : point_list) {
        i.neighboursx.clear();
        i.neighboursX.clear();
        i.neighbours2x.clear();
        i.neighbours2X.clear();
        i.NI.clear();
        i.NInII.clear();
        i.n1 = i.n2 = 0;

        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) && ((i.X - j.X).norm() < delta))
            {
                i.NI.push_back(j.Nr);
                i.neighboursx.push_back(j.x);
                i.neighboursX.push_back(j.X);
                i.n1++;
            }
        }

        for (size_t p = 0; p < i.NI.size(); ++p)
        {
            for (size_t q = 0; q < i.NI.size(); ++q)
            {
                if (i.NI[p] != i.NI[q])
                {
                    Eigen::Vector2d XiI = i.neighboursX[p] - i.X;
                    Eigen::Vector2d XiII = i.neighboursX[q] - i.X;

                    Eigen::Vector2d l = (XiI - XiII);

                    // 2D cross product (scalar determinant)
                    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
                    if ((std::abs(A) < 1e-6) && l.norm() < delta)
                    {
                        i.NInII.push_back(std::make_pair(i.NI[p], i.NI[q]));
                        i.neighbours2x.push_back(std::make_pair(point_list[i.NI[p]].x, point_list[i.NI[q]].x));
                        i.neighbours2X.push_back(std::make_pair(point_list[i.NI[p]].X, point_list[i.NI[q]].X));
                        i.n2++;
                    }
                }
            }
        }
    }
}

// Rest of functions (psifunc1, PP1, AA1, etc.) follow same pattern:
// 1. Change Vector3d→Vector2d
// 2. Convert 3D cross products→2D scalar determinants
// 3. Keep original variable names and code flow

double psifunc1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, const double C1)
{
    // Original energy calculation with 2D vectors
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;
    return C1 * s * s * 0.5;
}

double psifunc2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    return C2 * (1.0/3.0) * std::pow((std::abs(A) - std::abs(a))/std::abs(A), 2);
}


Eigen::Vector2d PP1(const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, const double C1)
{
    // Original force calculation with 2D vectors
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;
    Eigen::Vector2d eta = xiI.normalized();
    return C1 * eta * s;
}

Eigen::Vector2d PP2(const Eigen::Vector2d& XiI, const Eigen::Vector2d& XiII,const Eigen::Vector2d& xiI, const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    double G = (1/std::abs(A)) - (1/std::abs(a));

    Eigen::Vector2d H = xiII.squaredNorm() * xiI - xiII.dot(xiI) * xiII;
    return 2.0 * C2 * G * H;
}

Eigen::MatrixXd AA1(const int PD, const Eigen::Vector2d& XiI, const Eigen::Vector2d& xiI, const double C1)
{
    // Original stiffness calculation with 2D vectors
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;

    Eigen::Vector2d eta = xiI.normalized();
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd eta_dyad_eta = eta * eta.transpose();

    return C1 * ((s/l) * (II - eta_dyad_eta) + ((1/ L) * eta_dyad_eta));
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(int PD, const Eigen::Vector2d& XiI,const Eigen::Vector2d& XiII,const Eigen::Vector2d& xiI,const Eigen::Vector2d& xiII, int C2) {
    double A = XiI.x() * XiII.y() - XiI.y() * XiII.x();
    double a = xiI.x() * xiII.y() - xiI.y() * xiII.x();
    double AA = std::abs(A);
    double aa = std::abs(a);

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd BBI1 = (xiII.dot(xiII) * II) - (xiII * xiII.transpose());
    Eigen::MatrixXd BBJ1 = (2 * xiI * xiII.transpose()) - (xiI.dot(xiII) * II) - (xiII * xiI.transpose());

    Eigen::Vector2d eInII = (xiII.dot(xiII) * xiI - (xiI.dot(xiII) * xiII));
    Eigen::Vector2d eIInI = (xiI.dot(xiI) * xiII - (xiII.dot(xiI) * xiI));

    Eigen::MatrixXd AA2I = (2 * C2 * ((1/AA) - (1/aa)) * BBI1) + (2 * C2 * (1/std::pow(aa,3)) * (eInII * eInII.transpose()));

    Eigen::MatrixXd AA2J = (2 * C2 * ((1/AA) - (1/aa)) * BBJ1) + (2 * C2 * (1/std::pow(aa,3)) * (eInII * eIInI.transpose()));

    return {AA2I, AA2J};
}

// calculate_rk, assembly, update_points follow same conversion pattern:
// - Vector3d→Vector2d
// - 3D-specific math→2D equivalents
// - Keep all original debug statements and control flow
// [Previous code remains identical until calculate_rk]

void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD)
{
    //std::cout << "entered calculate_rk" << std::endl;
    constexpr double pi = 3.14159265358979323846;
    double Vh = pi * std::pow(delta, 2); // Area for 2D

    for (auto& i : point_list)
    {
        // Reset values
        i.residual = Eigen::Vector2d::Zero();
        i.R1 = Eigen::Vector2d::Zero();
        i.R2 = Eigen::Vector2d::Zero();
        i.psi = 0;

        //std::cout << "initialised 0's for R and psi" << std::endl;

        double JI = (i.n1 > 0) ? Vh / i.n1 : 0;
        double JInII = (i.n2 > 0) ? (Vh * Vh) / i.n2 : 0;

        //std::cout << "calculated JI and JInII" << std::endl;

        // Extended neighbor lists including self
        std::vector<int> neighboursEI = i.NI;
        std::vector<std::pair<int, int>> neighboursEII = i.NInII;
        std::vector<Eigen::Vector2d> neighboursEx = i.neighboursx;
        std::vector<Eigen::Vector2d> neighboursEX = i.neighboursX;

        neighboursEI.push_back(i.Nr);
        neighboursEII.push_back(std::make_pair(i.Nr, i.Nr));
        neighboursEx.push_back(i.x);
        neighboursEX.push_back(i.X);

        const int NNgbrEI = neighboursEI.size();
        const int NNgbrEII = neighboursEII.size();
        int total_cols = NNgbrEI + 2 * NNgbrEII;

        i.stiffness = Eigen::MatrixXd::Zero(PD * PD, total_cols);
        i.K1 = Eigen::MatrixXd::Zero(PD * PD, total_cols);
        i.K2 = Eigen::MatrixXd::Zero(PD * PD, total_cols);

        //std::cout << "initialised K1, K2, stiffness with 0's" << std::endl;

        // 1-neighbor interactions
        for (size_t j = 0; j < i.n1; j++) {
            Eigen::Vector2d XiI = i.neighboursX[j] - i.X;
            Eigen::Vector2d xiI = i.neighboursx[j] - i.x;

            //std::cout << "calculated XiI and xiI for 1-neighbor" << std::endl;

            auto psi1 = psifunc1(XiI, xiI, C1);
            Eigen::Vector2d R1_temp = PP1(XiI, xiI, C1);

            i.psi += JI * psi1;
            i.R1 += JI * R1_temp;

            //std::cout << "updated psi and residual for 1-neighbor" << std::endl;

            for (int b = 0; b < NNgbrEI; b++) {
                double K_factor = 0.0;
                if (i.NI[j] == neighboursEI[b]) K_factor += 1.0;
                if (i.Nr == neighboursEI[b]) K_factor -= 1.0;

                //std::cout << "K_factor: " << K_factor << std::endl;

                Eigen::MatrixXd AA1I = AA1(PD, XiI, xiI, C1);
                Eigen::MatrixXd stiffness_contribution = AA1I * JI * K_factor;

                i.K1.block(0, b, PD * PD, 1) =
                    Eigen::Map<Eigen::VectorXd>(stiffness_contribution.data(), PD * PD);
            }
        }

        // 2-neighbor interactions
        for (size_t j = 0; j < i.NInII.size(); j++) {
            Eigen::Vector2d XiI = i.neighbours2X[j].first - i.X;
            Eigen::Vector2d xiI = i.neighbours2x[j].first - i.x;
            Eigen::Vector2d XiII = i.neighbours2X[j].second - i.X;
            Eigen::Vector2d xiII = i.neighbours2x[j].second - i.x;

            //std::cout << "calculated XiI, XiII, xiI, xiII for 2-neighbor" << std::endl;

            auto psi2 = psifunc2(XiI, XiII, xiI, xiII, C2);
            Eigen::Vector2d R2_temp = PP2(XiI, XiII, xiI, xiII, C2);

            i.psi += JInII * psi2;
            i.R2 += JInII * R2_temp;

            //std::cout << "updated psi and residual for 2-neighbor" << std::endl;

            for (int b = 0; b < NNgbrEII; b++) {
                double K_factor_i = 0.0;
                if (i.NInII[j].first == neighboursEII[b].first) K_factor_i += 1.0;
                if (i.Nr == neighboursEII[b].first) K_factor_i -= 1.0;

                double K_factor_j = 0.0;
                if (i.NInII[j].second == neighboursEII[b].second) K_factor_j += 1.0;
                if (i.Nr == neighboursEII[b].second) K_factor_j -= 1.0;

                //std::cout << "K_factors: " << K_factor_i << ", " << K_factor_j << std::endl;

                auto [AA2I, AA2J] = AA2(PD, XiI, XiII, xiI, xiII, C2);
                Eigen::MatrixXd stiff_i = AA2I * K_factor_i * JInII;
                Eigen::MatrixXd stiff_j = AA2J * K_factor_j * JInII;

                i.K2.block(0, NNgbrEI + 2*b, PD*PD, 1) = Eigen::Map<Eigen::VectorXd>(stiff_i.data(), PD*PD);
                i.K2.block(0, NNgbrEI + 2*b + 1, PD*PD, 1) = Eigen::Map<Eigen::VectorXd>(stiff_j.data(), PD*PD);
            }
        }

        i.residual = i.R1 + i.R2;
        i.stiffness = i.K1 + i.K2;
        //std::cout << "final residual and stiffness calculated" << std::endl;
    }
}

void assembly(const int PD, const std::vector<Points>& point_list, int DOFs,
             Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag)
{
    if (flag == "residual") {
        R = Eigen::VectorXd::Zero(DOFs);
        for (const auto& point : point_list) {
            for (int d = 0; d < PD; d++) {
                if (point.BCflag[d] == 1 && point.DOF[d] > 0 && point.DOF[d] <= DOFs) {
                    const int index = static_cast<int>(point.DOF[d]) - 1;
                    if (index >= 0 && index < R.size()) {
                        R(index) += point.residual(d);
                    }
                }
            }
        }
    }
    else if (flag == "stiffness") {
        std::vector<Eigen::Triplet<double>> triplets;

        for (const auto& point : point_list) {
            const auto& stiffness = point.stiffness;
            const auto& nbrs_1 = point.NI;      // 1-neighbor list
            const auto& pairs_2 = point.NInII;  // 2-neighbor pairs

            // Create extended neighbor list (1-neighbors + self)
            std::vector<int> extended_nbrs = nbrs_1;
            extended_nbrs.push_back(point.Nr);  // MATLAB-style self-inclusion

            // Column indices mapping:
            // [0...n1-1] -> 1-neighbors
            // [n1]       -> self
            // [n1+1...]  -> 2-neighbor pairs (2 columns per pair)

            // Process 1-neighbor interactions (including self)
            for (size_t col = 0; col < extended_nbrs.size(); col++) {
                const int q = extended_nbrs[col];  // Neighbor index
                if (q < 0 || q >= point_list.size()) continue;

                // Extract PDxPD block for this neighbor column
                const Eigen::MatrixXd K_block = Eigen::Map<const Eigen::MatrixXd>(
                    stiffness.col(col).data(), PD, PD);

                // Add contributions to global matrix
                for (int i = 0; i < PD; i++) {
                    if (point.BCflag[i] != 1 || point.DOF[i] <= 0) continue;
                    const int row = point.DOF[i] - 1;

                    const auto& nbr_point = point_list[q];
                    for (int j = 0; j < PD; j++) {
                        if (nbr_point.BCflag[j] == 1 && nbr_point.DOF[j] > 0) {
                            const int col_idx = nbr_point.DOF[j] - 1;
                            triplets.emplace_back(row, col_idx, K_block(i,j));
                        }
                    }
                }
            }

            // Process 2-neighbor pair interactions
            for (size_t pair_idx = 0; pair_idx < pairs_2.size(); pair_idx++) {
                const auto& pair = pairs_2[pair_idx];
                const int q = pair.first;   // First neighbor in pair
                const int r = pair.second;  // Second neighbor in pair

                // Pair columns are offset after 1-neighbor columns
                const int col_q = extended_nbrs.size() + 2*pair_idx;
                const int col_r = col_q + 1;

                // Extract stiffness blocks for both pair members
                const Eigen::MatrixXd K_block_q = Eigen::Map<const Eigen::MatrixXd>(
                    stiffness.col(col_q).data(), PD, PD);
                const Eigen::MatrixXd K_block_r = Eigen::Map<const Eigen::MatrixXd>(
                    stiffness.col(col_r).data(), PD, PD);

                // Add contributions for first pair member (q)
                if (q >= 0 && q < point_list.size()) {
                    for (int i = 0; i < PD; i++) {
                        if (point.BCflag[i] != 1 || point.DOF[i] <= 0) continue;
                        const int row = point.DOF[i] - 1;

                        const auto& nbr_point = point_list[q];
                        for (int j = 0; j < PD; j++) {
                            if (nbr_point.BCflag[j] == 1 && nbr_point.DOF[j] > 0) {
                                const int col_idx = nbr_point.DOF[j] - 1;
                                triplets.emplace_back(row, col_idx, K_block_q(i,j));
                            }
                        }
                    }
                }

                // Add contributions for second pair member (r)
                if (r >= 0 && r < point_list.size()) {
                    for (int i = 0; i < PD; i++) {
                        if (point.BCflag[i] != 1 || point.DOF[i] <= 0) continue;
                        const int row = point.DOF[i] - 1;

                        const auto& nbr_point = point_list[r];
                        for (int j = 0; j < PD; j++) {
                            if (nbr_point.BCflag[j] == 1 && nbr_point.DOF[j] > 0) {
                                const int col_idx = nbr_point.DOF[j] - 1;
                                triplets.emplace_back(row, col_idx, K_block_r(i,j));
                            }
                        }
                    }
                }
            }
        }

        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::MatrixXd A = Eigen::MatrixXd(K);
        std::cout<<"Assembled global stiffness matrix: "<<std::endl<<A<<std::endl;
        std::cout<<"Assembled global residual vector: "<<std::endl<<R.transpose()<<std::endl;
    }
}

void update_points(const int PD, std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag)
{
    if (Update_flag == "Prescribed") {
        for (auto& i : point_list) {
            for (int p = 0; p < PD; p++) {
                if (i.BCflag[p] == 0) {
                    i.x[p] += i.X[p] + (LF * i.BCval[p]);
                    //std::cout << "Updated prescribed point " << i.Nr << " coord " << p << " to " << i.x[p] << std::endl;
                }
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (auto& i : point_list) {
            for (int q = 0; q < PD; q++) {
                if (i.BCflag[q] == 1 && i.DOF[q] > 0 && i.DOF[q] <= dx.size()) {
                    i.x[q] += dx[i.DOF[q] - 1];
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
            if (idx1 >= 0 && idx1 < point_list.size() &&
                idx2 >= 0 && idx2 < point_list.size()) {
                point.neighbours2x[i] = std::make_pair( point_list[idx1].x, point_list[idx2].x);
                }
        }
    }
}