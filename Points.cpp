#include "Points.h"
#include <iostream>
#include <cmath>
#include <Eigen/Sparse>

// Default constructor for the Points class
Points::Points()
    : Nr(0),                           // Initializing integer values
      X(Eigen::Vector3d::Zero()),      // Eigen vectors initialized to size 0
      x(Eigen::Vector3d::Zero()),      // Same for 'x'
      NI(),                            // Empty neighbor interaction list
      NInII(),                         // Empty 2-neighbor interaction list
      neighborsx(),                    // Empty list of neighbors' current coordinates
      neighborsX(),                    // Empty list of neighbors' reference coordinates
      neighbors2x(),                   // Empty list of 2-neighbor interactions
      neighbors2X(),                   // Empty list of 2-neighbor interactions
      Flag(""),                        // Default empty string
      BCflag(Eigen::Vector3d::Zero()), // Empty BCflag vector
      BCval(Eigen::Vector3d::Zero()),  // Empty BCval vector
      DOF(Eigen::Vector3d::Zero()),    // Empty DOF vector
      DOC(Eigen::Vector3d::Zero()),    // Empty DOC vector
      n1(0),                           // Initialize 1-neighbor count
      n2(0),                           // Initialize 2-neighbor count
      volume(0.0),                     // Initialize volume
      psi(0.0),                        // Initialize energy
      R1(),     // Residual for 1-neighbor (fixed to Vector 3d)
      R2(),     // Residual for 2-neighbor (fixed to Vector 3d)
      residual(), // Initialize residual
      // Initialize with appropriate dimensions for 3D
      K1(),  // Empty stiffness matrix for 1-neighbor
      K2(),  // Empty stiffness matrix for 2-neighbor
      stiffness(), // Initialize stiffness matrix
      JI(0.0),                         // 1-neighbor effective volume
      JII(0.0)                         // 2-neighbor effective volume
{}

std::vector<Points> mesh(double domain_size, int number_of_patches, double Delta, int number_of_right_patches, int& DOFs, int& DOCs, double d, const std::string& DEFflag, int PD)
{
    std::vector<Points> point_list;
    const int number_of_points = std::floor(domain_size / Delta) + 1;
    int total_points = number_of_patches + number_of_right_patches + number_of_points;
    int index = 0;

    Eigen::Matrix3d FF = Eigen::Matrix3d::Identity();

    if (DEFflag == "EXT") {
        FF(0, 0) = 1 + d;
    }
    else if (DEFflag == "EXP") {
        FF = (1 + d) * Eigen::Matrix3d::Identity();
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
            point.X = {(Delta / 2) + i * Delta, (Delta / 2) + j * Delta , 0};
            point.x = point.X;

            if (point.X[0] < (number_of_patches * Delta))
            {
                point.Flag = "Patch";
                point.BCval = {0, 0, 0};
                point.BCflag = {0, 0, 0};
                point.DOC[0] = ++DOCs;
            }
            else if ((point.X[0] > (Delta * (number_of_points + number_of_patches))))
            {
                point.Flag = "RightPatch";
                point.BCval = {d, 0, 0};
                point.BCflag = {0, 0, 0};
                point.DOC[0] = ++DOCs;
            }
            else
            {
                point.Flag = "Point";
                point.BCflag = {1, 0, 0};
                point.BCval = (FF * point.X) - point.X;
                point.DOF[0] = ++DOFs;
            }

            point.volume = 1;
            point_list.push_back(point);
        }
    }

    // Recalculate DOFs and assign correct indices
    DOFs = 0;
    for (auto& point : point_list) {
        if (point.BCflag[0] == 1) {
            point.DOF[0] = ++DOFs;
        }
    }

    return point_list;
}

// Neighbour list calculation
void neighbour_list(std::vector<Points>& point_list, double& delta)
{
    for (auto &i : point_list) {
        i.neighborsx.clear();
        i.neighborsX.clear();
        i.neighbors2x.clear();
        i.neighbors2X.clear();
        i.NI.clear();        // Clear the neighbor list
        i.NInII.clear();     // Clear the 2-neighbor list
        i.n1 = i.n2 = 0;

        for (auto &j : point_list) {
            if ((i.Nr != j.Nr) && ((i.X - j.X).norm() < delta))
            {
                i.NI.push_back(j.Nr);
                i.neighborsx.push_back(j.x);
                i.neighborsX.push_back(j.X);
                i.n1++;
            }
        }

        for (size_t p = 0; p < i.NI.size(); ++p)
        {
            for (size_t q = 0; q < i.NI.size(); ++q)
            {
                if (i.NI[p] != i.NI[q])
                {
                    Eigen::Vector3d XiI = i.neighborsX[p] - i.X;
                    Eigen::Vector3d XiII = i.neighborsX[q] - i.X;

                    Eigen::Vector3d l = (XiI - XiII);

                    // Using norm() instead of cross product for better type safety
                    Eigen::Vector3d A = XiI.cross(XiII);
                    if ((A.norm() < 1e-6) && l.norm() < delta)
                    {
                        i.NInII.push_back(std::make_pair(i.NI[p], i.NI[q]));
                        i.neighbors2x.push_back(std::make_pair(point_list[i.NI[p]].x, point_list[i.NI[q]].x));
                        i.neighbors2X.push_back(std::make_pair(point_list[i.NI[p]].X, point_list[i.NI[q]].X));
                        i.n2++;  // Increment here instead of outside the loop
                    }
                }
            }
        }
    }
}

// Energy function for 1-neighbor interaction
double psifunc1(const Eigen::Vector3d& XiI, const Eigen::Vector3d& xiI, const double C1)
{
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;

    return C1 * s * s * 0.5;
}

// Energy function for 2-neighbor interaction
double psifunc2(const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const int C2)
{
    Eigen::Vector3d A = XiI.cross(XiII);
    Eigen::Vector3d a = xiI.cross(xiII);

    double s = ((A.norm() - a.norm()) / A.norm());
    return C2 * s * s * (1.0/3.0);
}

// Force calculation for 1-neighbor interaction
Eigen::Vector3d PP1(const Eigen::Vector3d& XiI, const Eigen::Vector3d& xiI, const double C1)
{
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;
    Eigen::Vector3d eta = xiI / l;
    return C1 * eta * s;
}

// Force calculation for 2-neighbor interaction
Eigen::Vector3d PP2(const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const int C2)
{
    Eigen::Vector3d A = XiI.cross(XiII);
    Eigen::Vector3d a = xiI.cross(xiII);

    double G = (1.0 / A.norm()) - (1.0 / a.norm());

    double xiII_sq = xiII.dot(xiII);       // xiII' * xiII
    double xiII_dot_xiI = xiII.dot(xiI);   // xiII' * xiI

    Eigen::Vector3d H = xiII_sq * xiI - xiII_dot_xiI * xiII;

    return 2.0 * C2 * G * H;
}

// Stiffness calculation for 1-neighbor interaction
Eigen::MatrixXd AA1(const int PD, const Eigen::Vector3d& XiI, const Eigen::Vector3d& xiI, const double C1)
{
    double L = XiI.norm();
    double l = xiI.norm();
    double s = (l - L) / L;

    Eigen::VectorXd eta = xiI / l;
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    Eigen::MatrixXd eta_dyad_eta = eta * eta.transpose();

    return C1 * ((s/l) * (II- eta_dyad_eta) + ((1/ L) * eta_dyad_eta));
}

// Stiffness calculation for 2-neighbor interaction
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(const int PD, const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const int C2)
{
    Eigen::Vector3d A = XiI.cross(XiII);
    Eigen::Vector3d a = xiI.cross(xiII);

    double AA = A.norm();
    double aa = a.norm();

    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD,PD);
    Eigen::MatrixXd BBI1 = ((xiII.dot(xiII) * II) - (xiII * xiII.transpose()));
    Eigen::MatrixXd BBJ1 = ((2 * xiI * xiII.transpose()) - (xiI.dot(xiII) * II) - (xiII * xiI.transpose()));

    Eigen::VectorXd eInII = ((xiII.dot(xiII)) * xiI - (xiI.dot(xiII)) * xiII);
    Eigen::VectorXd eIInI = ((xiI.dot(xiI)) * xiII - (xiII.dot(xiI)) * xiI);

    Eigen::MatrixXd AA2I = ((2 * C2 * ((1/AA) - (1/aa)) * BBI1) + (2 * C2 * ((1/std::pow(aa,3))) * (eInII * eInII.transpose())));
    Eigen::MatrixXd AA2J = ((2 * C2 * ((1/AA) - (1/aa)) * BBJ1) + (2 * C2 * ((1/std::pow(aa,3))) * (eInII * eIInI.transpose())));

    return {AA2I, AA2J};
}

// Calculate tangent stiffness and energy
// Calculate tangent stiffness and energy
void calculate_rk(std::vector<Points>& point_list, double C1, double C2, double delta, int PD)
{
    constexpr double pi = 3.14159265358979323846;
    double Vh = (4.0 / 3.0) * pi * std::pow(delta, 3);

    for (auto& i : point_list)
    {
        // Reset values to proper dimensions
        i.residual = Eigen::Vector3d::Zero();
        i.R1 = Eigen::Vector3d::Zero();
        i.R2 = Eigen::Vector3d::Zero();
        i.psi = 0;

        double JI = Vh / i.n1;
        double JInII = (Vh * Vh) / i.n2;

        // Create extended neighbor list (including the point itself)
        std::vector<int> neighborsEI = i.NI;
        std::vector<std::pair<int, int>> neighborsEII = i.NInII;

        std::vector<Eigen::Vector3d> neighborsEx = i.neighborsx;
        std::vector<Eigen::Vector3d> neighborsEX = i.neighborsX;

        // Add the point itself to the extended neighbors
        neighborsEI.push_back(i.Nr);
        neighborsEII.push_back(std::make_pair(i.Nr,i.Nr));
        neighborsEx.push_back(i.x);
        neighborsEX.push_back(i.X);

        const int NNgbrEI = neighborsEI.size();
        const int NNgbrEII = neighborsEII.size();

        // Create a unified column dimension for K1 and K2
        //int unified_cols = NNgbrEI + NNgbrEII;

        // Initialize matrices with consistent dimensions
        i.K1 = Eigen::MatrixXd::Zero(PD * PD, NNgbrEII);
        i.K2 = Eigen::MatrixXd::Zero(PD * PD, NNgbrEII);
        i.stiffness = Eigen::MatrixXd::Zero(PD * PD, NNgbrEII);

        for (size_t j = 0; j < i.n1; j++) {
            Eigen::VectorXd XiI = i.neighborsX[j] - i.X;
            Eigen::VectorXd xiI = i.neighborsx[j] - i.x;

            auto psi1 = psifunc1(XiI, xiI, C1);
            Eigen::VectorXd R1_temp = PP1(XiI, xiI, C1);
            // Calculate energy and residual
            i.psi += JI * psi1;
            i.R1 += JI * R1_temp;

            // Calculate stiffness for each neighbor including self
            for (int b = 0; b < NNgbrEI; b++)
            {
                // This implements the (neighbors(i)==neighborsEI(b))-(a==neighborsEI(b)) logic
                double K_factor = 0.0;
                if (i.NI[j] == neighborsEI[b])
                {
                    K_factor += 1.0;
                }
                if (i.Nr == neighborsEI[b])
                {
                    K_factor -= 1.0;
                }

                // For 1D, the AA1 function simplifies to C1/L
                Eigen::MatrixXd AA1I = AA1(PD, XiI, xiI, C1);
                Eigen::MatrixXd stiffness_contribution = AA1I * JI * K_factor;
                i.K1.block(0, b, PD * PD, 1) = Eigen::Map<Eigen::VectorXd>(stiffness_contribution.data(), PD * PD);
            }
        }

        for (size_t j = 0; j < i.NInII.size(); j++)
        {
            Eigen::VectorXd XiI = (i.neighbors2X[j]).first - i.X;
            Eigen::VectorXd xiI = (i.neighbors2x[j]).first - i.x;

            Eigen::VectorXd XiII = (i.neighbors2X[j]).second - i.X;
            Eigen::VectorXd xiII = (i.neighbors2x[j]).second - i.x;

            auto psi2 = psifunc2(XiI,XiII,xiI,xiII,C2);
            Eigen::VectorXd R2_temp = PP2(XiI,XiII,xiI,xiII,C2);

            i.psi += JInII * psi2;
            i.R2 += JInII * R2_temp;

            for (int b = 0; b < NNgbrEII; b++)
            {
                // This implements the (neighbors(i)==neighborsEI(b))-(a==neighborsEI(b)) logic
                double K_factor_i = 0.0;
                if ((i.NInII[j]).first == (neighborsEII[b]).first) {
                    K_factor_i += 1.0;
                }
                if (i.Nr == (neighborsEII[b]).first) {
                    K_factor_i -= 1.0;
                }

                // Check for the condition involving j (neighbors(j) == neighborsE(b) and a == neighborsE(b))
                double K_factor_j = 0.0;
                if ((i.NInII[j]).second == (neighborsEII[b]).second) {
                    K_factor_j += 1.0;
                }
                if (i.Nr == (neighborsEII[b]).second) {
                    K_factor_j -= 1.0;
                }

                Eigen::MatrixXd AA2I = AA2(PD, XiI, XiII, xiI,xiII,C2).first;
                Eigen::MatrixXd AA2J = AA2(PD, XiI, XiII, xiI,xiII,C2).second;

                Eigen::MatrixXd stiffness_contribution_i = AA2I * K_factor_i;
                Eigen::MatrixXd stiffness_contribution_j = AA2J * K_factor_j;

                // For 1D, the AA1 function simplifies to C1/L
                Eigen::MatrixXd stiffness_contribution = JInII * (stiffness_contribution_i + stiffness_contribution_j);
                i.K2.block(0, b, PD * PD, 1) = Eigen::Map<Eigen::VectorXd>(stiffness_contribution.data(), PD * PD);
            }
        }
        i.residual = i.R1 + i.R2;
        i.stiffness = i.K1 + i.K2;
    }
}

// Assemble residual or stiffness matrix
void assembly(const int PD, const std::vector<Points>& point_list, int DOFs, Eigen::VectorXd& R, Eigen::SparseMatrix<double>& K, const std::string& flag)
{
    if (flag == "residual") {
        // Reset residual vector
        R = Eigen::VectorXd::Zero(DOFs);

        // Assemble residual
        for (const auto& point : point_list) {
            Eigen::Vector3d R_P = point.residual;
            Eigen::Vector3d BCflg = point.BCflag;
            Eigen::Vector3d DOF = point.DOF;

            for (int ii = 0; ii < PD; ii++)
            {
                if (BCflg[ii] == 1 && DOF[ii] > 0 && DOF[ii] <= DOFs) {
                    R[((DOF[ii]) - 1)] += R_P[ii]; // Add specific component
                    //std::cout << "[Residual] Added residual " << R_P[ii] << " at DOF " << DOF[ii] << " (Global Point ID: " << point.Nr << ")\n";
                }
            }
        }

        //std::cout << "Size of the residual vector is: " << R.size() << "\n";
        //std::cout << "\nResidual Vector R:\n" << R << std::endl;
    }
    else if (flag == "stiffness") {
        // Reset stiffness matrix

        K.setZero();
        std::vector<Eigen::Triplet<double>> triplets;

        // Assemble stiffness
        for (const auto& point : point_list) {
            Eigen::MatrixXd K_P = point.stiffness;
            Eigen::Vector3d BCflg_p = point.BCflag;
            Eigen::Vector3d DOF_p = point.DOF;

            for (int pp = 0; pp < PD; pp++)
            {
                if (BCflg_p[pp] == 1 && DOF_p[pp] > 0 && DOF_p[pp] <= DOFs) {
                    // Create extended neighbor list including the point itself
                    std::vector<int> neighborsEI = point.NI;
                    neighborsEI.push_back(point.Nr);

                    // Process each neighbor
                    for (size_t q = 0; q < neighborsEI.size(); q++) {
                        int nbr_idx = neighborsEI[q];
                        if (nbr_idx >= 0 && nbr_idx < (point_list.size())) {
                            // Extract the stiffness block for this neighbor
                            Eigen::MatrixXd K_PQ = K_P.block(0, q*PD, PD, PD);

                            Eigen::Vector3d BCflg_q = point_list[nbr_idx].BCflag;
                            Eigen::Vector3d DOF_q = point_list[nbr_idx].DOF;

                            for(int qq = 0; qq < PD; qq++)
                            {
                                if (BCflg_q[qq] == 1 && DOF_q[qq] > 0 && DOF_q[qq] <= DOFs) {
                                    double Kval = K_PQ(pp, qq);
                                    triplets.emplace_back(((DOF_p[pp]) - 1), ((DOF_q[qq]) - 1), Kval);
                                    //std::cout << "[Stiffness] K(" << DOF_p[pp] << "," << DOF_q[qq] << ") = " << Kval << " from Point " << point.Nr << " to Point " << nbr_idx << "\n";
                                }
                            }
                        }
                    }
                }
            }
        }
        K.resize(DOFs, DOFs);
        K.setFromTriplets(triplets.begin(), triplets.end());

        // Convert to dense for display
        Eigen::MatrixXd A = Eigen::MatrixXd(K);

        //std::cout << "\nStiffness matrix size: " << A.rows() << " x " << K.cols() << std::endl;
        //std::cout << "\nStiffness Matrix K:\n" << A << std::endl;
    }
}

// Update points based on displacement or prescribed values
void update_points(const int PD, std::vector<Points>& point_list, double LF, Eigen::VectorXd& dx, const std::string& Update_flag)
{
    if (Update_flag == "Prescribed") {
        for (auto& i : point_list)
        {
            for (size_t p = 0; p < PD; p++)
            {
                if (i.BCflag[p] == 0)
                {
                    i.x = i.x + (LF * i.BCval);
                }
            }
        }
    }
    else if (Update_flag == "Displacement") {
        for (auto& i : point_list) {
            for (size_t q = 0; q < PD; q++)
            {
                if (i.BCflag[q] == 1 && i.DOF[q] > 0 &&
                    (i.DOF[q]) <= dx.size())
                {
                    i.x[q] += dx[i.DOF[q] - 1];
                }
            }
        }
    }

    // Update neighbor coordinates after points have been updated
    for (auto& point : point_list) {
        // Update 1-neighbor coordinates
        for (size_t i = 0; i < point.NI.size(); i++) {
            int nbr_idx = point.NI[i];
            if (nbr_idx >= 0 && nbr_idx < (point_list.size())) {
                point.neighborsx[i] = point_list[nbr_idx].x;
            }
        }

        // Update 2-neighbor coordinates
        for (size_t i = 0; i < point.NInII.size(); i++) {
            int idx1 = point.NInII[i].first;
            int idx2 = point.NInII[i].second;

            if (idx1 >= 0 && idx1 < (point_list.size()) &&
                idx2 >= 0 && idx2 < (point_list.size())) {
                point.neighbors2x[i] = std::make_pair(point_list[idx1].x, point_list[idx2].x);
            }
        }
    }
}