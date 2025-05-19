#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <algorithm>
#include "Points.h"

/*
Coversation column - Update discussion
current issue:
for C2 < C1, the code is smooth, its converign in 7 / 3 iteration depending on the order of differnece between C1 and C2.ADJ_OFFSET_SINGLESHOTbot 
if C2 >= C1, the code is not converging, moreover its blowing up so much the values are becoming nan/inf.
*/

// --- Main Function ---
int main() {

    // Parameters
    const int PD = 2;
    std::cout << "\n======================================================" << std::endl;
    std::cout << "Starting "<< PD <<"D Peridynamics Simulation" << std::endl;
    std::cout << "======================================================" << std::endl;
    double domain_size = 1.0;
    double delta = 0.301;
    double Delta = 0.10;
    double d = 1.0e-4;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 1.0e-2;
    double C2 = 1.0e-1;
    std::string DEFflag = "EXP";
    int DOFs = 0;
    int DOCs = 0;

    // Create mesh
    std::vector<Points> points = mesh(domain_size, number_of_patches, Delta, number_of_right_patches, DOFs, DOCs, d, DEFflag, PD);
    std::cout << "Mesh contains " << points.size() << " points with " << DOFs << " DOFs\n";
    neighbour_list(points, delta);

    // Debugging the points and their neighbours
/*
    for (const auto& i : points) {
        std::cout << "Nr: " << i.Nr << std::endl;
        std::cout << "X: [ " << i.X.transpose() << " ]" << std::endl;
        std::cout << "x: [ " << i.x.transpose() << " ]" << std::endl;
        std::cout << "Volume: " << i.volume << std::endl;
        std::cout << "BC Flag: [ " << i.BCflag.transpose() << " ]" << std::endl << "Flag: " << i.Flag << std::endl;
        std::cout << "BC Value: [ " << i.BCval.transpose() << " ]" << std::endl;
        std::cout << "1-Neighbours of " << i.Nr << " are: [";
        for (const auto& n : i.NI)
        {
            std::cout << "{ " << n << " " << "} ";
        }
        std::cout << "]";
        std::cout << "\nNumber of 1-neighbour interactions for point " << i.Nr << ": " << i.n1 << std::endl;
        std::cout << "2-Neighbours of " << i.Nr << " are: [";
        for (const auto& p : i.NInII)
        {
            std::cout << "{ " << p.first << ", " << p.second << " } ";
        }
        std::cout << std::endl;
        std::cout << "]";
        std::cout << "\nNumber of 2-neighbour interactions for point " << i.Nr << ": " << i.n2 << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
*/
    
    // Newton-Raphson setup
    double steps = 10.0;
    double load_step = (1.0 / steps);
    double tol = 1e-6;
    int max_try = 10;
    double LF = 0.0;

    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Delta: " << Delta<< " | Horizon: " << delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << " C2: " << C2 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Initialize Eigen objects
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);
    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd dx;
    std::cout<<"initialised K, R and dx"<<std::endl;

    
    // Load stepping loop
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;

        //std::cout<<"going into update points for displacing right patch"<<std::endl;
        // Apply prescribed displacements
        update_points(PD ,points, LF, dx, "Prescribed", delta, DOFs);
        //std::cout<<"coming out of update points after displacing right patch"<<std::endl;

        int error_counter = 1;
        bool isNotAccurate = true;
        double normnull = 0.0;
        //std::cout<<"going into newton raphson steps"<<std::endl;
        // Newton-Raphson iteration
        while (isNotAccurate && error_counter <= max_try) {
            //std::cout<<"going into calculate_rk for calculating elemental R and K"<<std::endl;
            calculate_rk(points, C1, C2, delta, PD);
            //std::cout<<"coming out of calculate_rk for calculating elemental R and K"<<std::endl;
            //std::cout<<"going into assemble for assembling global R "<<std::endl;
            assembly(PD,points, DOFs, R, K, "residual");
            //std::cout<<"coming out of assemble after assembling global R "<<std::endl;
            //std::cout<<"Residual: "<< std::endl << R.transpose() << std::endl;

            double residual_norm = R.norm();
            double rel_norm = 0.0;
            if (error_counter == 1) {
                normnull = residual_norm;
                std::cout << "Initial Residual Norm: " << residual_norm << std::endl;
            } else {
                rel_norm = (residual_norm == 0) ? 0 : (residual_norm / normnull);
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm
                          << ", Relative = " << rel_norm << std::endl;

                if (rel_norm < tol || residual_norm < tol) {
                    isNotAccurate = false;
                    std::cout << "Converged after " << error_counter << " iterations." << std::endl;
                }
            }
            //std::cout<<"going into assemble for assembling global K "<<std::endl;
            assembly(PD,points, DOFs, R, K, "stiffness");
            //std::cout<<"coming out of assemble after assembling global K "<<std::endl;
            
            //std::cout<<"Stiffness: "<< std::endl << K << std::endl;
            Eigen::MatrixXd Kdense = Eigen::MatrixXd(K);
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Kdense);
            double cond = svd.singularValues()(0) / svd.singularValues().tail(1)(0);
            std::cout << "Condition number of K: " << cond << std::endl;

            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            /*if (solver.info() != Eigen::Success) {
                std::cerr << "Solver decomposition failed!" << std::endl;
            }*/
            dx = solver.solve(-R);
            /*if (solver.info() != Eigen::Success) {
                std::cerr << "Solver failed to converge!" << std::endl;
                dx.setZero(); // Force dx to zero on failure
            }*/
            std::cout<<"dx: "<< dx.size()<< std::endl << dx.transpose() << std::endl;

            update_points(PD, points, LF, dx, "Displacement",delta, DOFs);
            error_counter++;
            std::cout<<std::endl;
        }

        LF += load_step;

        // Output current state
        for (const auto& p : points) {
            //std::cout << "Point " << p.Nr << " was at X: " << p.X.transpose() << " and moved to x: " << p.x.transpose() << ", displacement = " << (p.x[0] - p.X[0]) << std::endl;
        }
        //std::cout<<std::endl;
    }
        
    return 0;
}