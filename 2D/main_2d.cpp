// Main.cpp
// 1D Peridynamics Simulation Program
// This program implements a peridynamics-based solver for 1D problems using Newton-Raphson iteration

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
#include <chrono>

#include "Points_2d.h"
#include "cli.h"
#include "logger.h"

// ============================================================================
// DEBUG: Check neighbor assignments for all points
// ============================================================================
void debug_neighbor_assignments(const std::vector<Point>& PL, int PD) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "NEIGHBOR ASSIGNMENT DEBUG REPORT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    int total_points = PL.size();
    int points_with_self_in_neighbors = 0;
    int max_neighbors = 0;
    int min_neighbors = INT_MAX;
    double avg_neighbors = 0.0;
    
    // Check each point
    for (size_t i = 0; i < PL.size(); i++) {
        const auto& point = PL[i];
        
        // Check if neighbors list contains self
        bool has_self = false;
        for (int nbr : point.neighbors) {
            if (nbr == (int)i) {
                has_self = true;
                break;
            }
        }
        
        if (has_self) {
            points_with_self_in_neighbors++;
            std::cout << "\n*** WARNING: Point " << i << " (Nr=" << point.Nr 
                      << ") has SELF in neighbors list! ***" << std::endl;
        }
        
        // Track neighbor counts
        int num_neighbors = point.neighbors.size();
        max_neighbors = std::max(max_neighbors, num_neighbors);
        min_neighbors = std::min(min_neighbors, num_neighbors);
        avg_neighbors += num_neighbors;
        
        // Detailed debug for first 10 points and any with self
        if (i < 10 || has_self) {
            std::cout << "\nPoint " << i << " (Nr=" << point.Nr << "):" << std::endl;
            std::cout << "  Position: X = (" << point.X[0] << ", " << point.X[1];
            if (PD == 3) std::cout << ", " << point.X[2];
            std::cout << ")" << std::endl;
            
            std::cout << "  # neighbors: " << num_neighbors << std::endl;
            
            // Show first few neighbors
            int show_count = std::min(8, num_neighbors);
            std::cout << "  First " << show_count << " neighbors: ";
            for (int n = 0; n < show_count; n++) {
                std::cout << point.neighbors[n];
                if (point.neighbors[n] == (int)i) {  // i is the array index
    std::cout << "(SELF-ARRAY-INDEX)";
} else if (point.neighbors[n] < PL.size() && 
           PL[point.neighbors[n]].Nr == point.Nr) {
    std::cout << "(SELF-Nr-MATCH)";  // This should never happen!
}
                if (n < show_count - 1) std::cout << ", ";
            }
            if (num_neighbors > show_count) std::cout << " ... (+" << (num_neighbors - show_count) << " more)";
            std::cout << std::endl;
            
            // Check neighbor coordinates
            if (i < 5 && point.neighbors.size() > 0) {
                std::cout << "  Neighbor coordinate check for first neighbor:" << std::endl;
                int first_nbr_idx = point.neighbors[0];
                if (first_nbr_idx >= 0 && first_nbr_idx < (int)PL.size()) {
                    const auto& nbr_point = PL[first_nbr_idx];
                    std::cout << "    Neighbor " << first_nbr_idx << " (Nr=" << nbr_point.Nr 
                              << ") at X = (" << nbr_point.X[0] << ", " << nbr_point.X[1];
                    if (PD == 3) std::cout << ", " << nbr_point.X[2];
                    std::cout << ")" << std::endl;
                    
                    // Check if neighbor has reciprocal connection
                    bool reciprocal = false;
                    for (int nbr_nbr : nbr_point.neighbors) {
                        if (nbr_nbr == (int)i) {
                            reciprocal = true;
                            break;
                        }
                    }
                    std::cout << "    Reciprocal connection: " << (reciprocal ? "YES" : "NO") << std::endl;
                }
            }
        }
    }
    
    avg_neighbors /= total_points;
    
    // Summary statistics
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "NEIGHBOR ASSIGNMENT SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Total points: " << total_points << std::endl;
    std::cout << "Points with self in neighbors: " << points_with_self_in_neighbors 
              << " (" << (100.0 * points_with_self_in_neighbors / total_points) << "%)" << std::endl;
    std::cout << "Neighbor counts - Min: " << min_neighbors 
              << ", Max: " << max_neighbors 
              << ", Avg: " << avg_neighbors << std::endl;
    
    // Check for common issues
    std::cout << "\nCOMMON ISSUES CHECK:" << std::endl;
    std::cout << "1. Self in neighbors: ";
    if (points_with_self_in_neighbors > 0) {
        std::cout << "FAIL - " << points_with_self_in_neighbors << " points have self as neighbor" << std::endl;
        std::cout << "   -> This will cause duplicate self entries in stiffness calculation!" << std::endl;
    } else {
        std::cout << "PASS" << std::endl;
    }
    
    // Check if any point has unusually high/low neighbor count
    int unusual_count = 0;
    for (const auto& point : PL) {
        if (point.neighbors.size() < 5 || point.neighbors.size() > 50) {
            unusual_count++;
        }
    }
    std::cout << "2. Unusual neighbor counts (<5 or >50): " << unusual_count 
              << " points" << std::endl;
    
    // Check neighborX and neighborx vectors
    std::cout << "\nNEIGHBOR COORDINATE VECTORS CHECK:" << std::endl;
    int inconsistent_count = 0;
    for (size_t i = 0; i < std::min(5, (int)PL.size()); i++) {
        const auto& point = PL[i];
        if (point.neighbors.size() != point.neighborsX_vec.size() ||
            point.neighbors.size() != point.neighborsx_vec.size()) {
            inconsistent_count++;
            std::cout << "  Point " << i << ": neighbors=" << point.neighbors.size()
                      << ", neighborsX_vec=" << point.neighborsX_vec.size()
                      << ", neighborsx_vec=" << point.neighborsx_vec.size() << std::endl;
        }
    }
    if (inconsistent_count == 0) {
        std::cout << "  All coordinate vectors match neighbor counts ✓" << std::endl;
    }
    
    // Detailed check for points with self in neighbors
    if (points_with_self_in_neighbors > 0) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "DETAILED CHECK OF POINTS WITH SELF IN NEIGHBORS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        for (size_t i = 0; i < PL.size(); i++) {
            const auto& point = PL[i];
            bool has_self = false;
            int self_position = -1;
            
            for (size_t n = 0; n < point.neighbors.size(); n++) {
                if (point.neighbors[n] == point.Nr) {
                    has_self = true;
                    self_position = n;
                    break;
                }
            }
            
            if (has_self) {
                std::cout << "\nPoint " << i << " (Nr=" << point.Nr << "):" << std::endl;
                std::cout << "  Self at position " << self_position << " in neighbors list" << std::endl;
                std::cout << "  All neighbors: ";
                for (size_t n = 0; n < point.neighbors.size(); n++) {
                    std::cout << point.neighbors[n];
                    if (n == (size_t)self_position) std::cout << "(SELF)";
                    if (n < point.neighbors.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
                
                // Show positions of these neighbors
                if (point.neighbors.size() <= 10) {
                    std::cout << "  Neighbor positions:" << std::endl;
                    for (size_t n = 0; n < point.neighbors.size(); n++) {
                        int nbr_idx = point.neighbors[n];
                        if (nbr_idx >= 0 && nbr_idx < (int)PL.size()) {
                            const auto& nbr = PL[nbr_idx];
                            double dist = (nbr.X - point.X).norm();
                            std::cout << "    Neighbor " << nbr_idx << " (Nr=" << nbr.Nr 
                                      << "): dist = " << dist;
                            if (n == (size_t)self_position) std::cout << " (SELF - distance should be 0)";
                            std::cout << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "END NEIGHBOR DEBUG REPORT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Pause to read output
    std::cout << "\nPress Enter to continue..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}

// --- Main Function --- //
int main(int argc, char* argv[]) {
    // Start timing the entire program execution
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting 1D Peridynamics simulation!" << std::endl;

    
    /////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// SIMULATION SETUP///////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Parse command line arguments and initialize simulation parameters
    CLIOptions opts = parseArguments(argc, argv);
    int PD = 2;  // Problem dimension (1D)
    double domain_size = opts.domain_size;  // Total size of the computational domain
    double Delta = opts.Delta;  // Horizon radius (interaction distance)
    double L = opts.L;  // Lattice spacing / grid spacing
    double d = opts.d * domain_size;  // Damage parameter scaled by domain size
    int number_of_patches = opts.number_of_patches;  // Number of boundary patches on left
    int number_of_right_patches = opts.number_of_right_patches;  // Number of boundary patches on right
    double C1 = opts.C1;  // Material constant for peridynamic formulation
    double C2 = opts.C2;  // Second material constant (if applicable)
    double nn = opts.nn;  // Nonlocal parameter / power law exponent
    double F_prescribed = opts.F_prescribed;  // Prescribed force value
    std::string Prescribed_Flag = opts.Prescribed_Flag;  // Type of boundary condition ("Force" or "Displacement")
    std::string DEFflag = opts.DEFflag;  // Deformation flag
    std::string file_name = opts.output_dir;  // Output file name for logging
    int DOFs;  // Total number of degrees of freedom (to be computed)


    std::cout<<"begin the simulation with parameters: "<< std::endl;
    std::cout<<"Domain Size: "<< domain_size << " | Lattice Length / Delta: "<< L << " | Horizon: "<< Delta << std::endl;

    // 1. Compute the domain corners (boundaries)
    auto Corners = Compute_Corners(PD, domain_size);

    std::cout<<"Corners computed."<< std::endl;
    // 2. Create the interior mesh discretization
    auto mesh_result = Mesh(PD, Corners, L);
std::vector<Eigen::Vector3d> NLtmp = mesh_result.first;
std::vector<std::vector<int>> CNCT = mesh_result.second;
    
    
    std::cout<<"Interior mesh created with "<< NLtmp.size() <<" nodes."<< std::endl;
    // Create external patch nodes at boundaries
    std::vector<Eigen::Vector3d> NLext = Patch(PD, Corners, L, Delta, number_of_patches, number_of_right_patches);

    // Combine interior mesh and boundary patches into single node list
    std::vector<Eigen::Vector3d> NL = NLtmp;
    NL.insert(NL.end(), NLext.begin(), NLext.end());
    
    // Sort all nodes in ascending order of position
    // std::sort(NL.begin(), NL.end(), [](const double& a, const double& b) {
    //     return a < b;
    // });

    std::string TOPflag = "FULL";
    std::vector<double> Bvals;
    // 3. Create point topology (convert node positions to Point objects with properties)
    auto [PL, EL] = Topology(PD, NL,CNCT,  L, Delta, Bvals, TOPflag);

    std::cout<<"Point topology created with "<< PL.size() <<" points and "<< EL.size() <<" elements."<< std::endl;
    // 4. Assign neighbor relationships (which points interact with each other within horizon)
    PL = AssignNgbrs(PD,  PL, L, Delta);

    std::cout<<"Neighbors assigned to points."<< std::endl;
    // 5. Assign volumes to each point for integration
    PL = AssignVols( Corners, PL, L, Bvals, TOPflag);

    std::cout<<"Volumes assigned to points."<< std::endl;

    
    // Validate generated points for correctness
    
    // 6. Output mesh statistics
    // std::cout << "======================================================" << std::endl;
    // std::cout << "number of nodes                 : " << NL.size() << std::endl;
    // std::cout << "number of points                : " << PL.size() << std::endl;

    // 7. Compute FF (force factor) for boundary conditions
    Eigen::Matrix3d FF = Compute_FF(PD, d, DEFflag);
    
    std::cout<<"Deformation gradient FF computed."<< std::endl;
    // 8. Assign boundary conditions to points
    std::string BCflag = "STD";        // your code supports "STD" / "DBC"
    std::string PatchFlag = "fullpatch"; // "fullpatch" / "horzpatch" / "vertpatch"

    auto bc_result = AssignBCs(Corners, PL, FF, BCflag, PatchFlag);
    PL   = bc_result.first;
    DOFs = bc_result.second;  // NOTE: AssignBCs already calls AssignGlobalDOF inside it
    std::cout << "Boundaries assigned to points." << std::endl;
    // Assign global degree of freedom numbering to free points
    auto result = AssignGlobalDOF(PL);
    PL = result.first;
    DOFs = result.second;  // Total number of DOFs in the system
    std::cout << "number of DOFs                  : " << DOFs << std::endl;
    std::cout << "======================================================" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////    LOGGING THE INFO     ////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Initialize logger for writing simulation data to log files
    std::cout << "[LOG] Writing to: log_files/" << file_name << ".log" << std::endl;
    Logger logger(file_name);
    logger.writeHeader(file_name);
    
    // Prepare CSV file for timing results (append mode, with header if new)
    std::string timing_csv = "csv_files/timing_results.csv";
    bool file_exists = std::filesystem::exists(timing_csv);
    std::ofstream csv_file(timing_csv, std::ios::app);
    if (!file_exists) {
        // Write header only if file doesn't exist
        csv_file << "spacing,number_of_points,simulation_time_sec,total_time_sec,implementation\n";
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// NEWTON - RAPHSON SOLVER //////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Newton-Raphson solver parameters
    int steps = opts.steps;  // Number of load steps
    double load_step = 1.0 / steps;  // Incremental load factor per step
    double tol = opts.tol;  // Convergence tolerance
    int max_try = 100;  // Maximum number of Newton-Raphson iterations per load step
    double LF = 0.0;  // Current load factor (starts at 0, goes to 1)
    Eigen::Vector3d F_rec_patch = Eigen::Vector3d::Zero();
    Eigen::Vector3d F_rec_rightpatch = Eigen::Vector3d::Zero();

    //debug_neighbor_assignments(PL, PD);
    // Write simulation parameters to log file
    logger.writeParameters(domain_size, L, Delta, PL.size(), steps, C1, nn, Prescribed_Flag, F_prescribed, d, number_of_patches, number_of_right_patches);

    // Display simulation parameters to console
    std::cout << "======================================================" << std::endl;
    std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Domain Size: " << domain_size << " | Lattice Length / Delta: " << L<< " | Horizon: " << Delta << std::endl;
    std::cout << "Steps: " << steps << " | Load Step: " << load_step<< " | Tolerance: " << tol << std::endl;
    std::cout << "Material constant C1: " << C1 << std::endl;
    std::cout << "======================================================" << std::endl;

    // Initialize Eigen linear algebra objects for the system
    Eigen::VectorXd R = Eigen::VectorXd::Zero(DOFs);  // Residual vector
    Eigen::SparseMatrix<double> K;  // Stiffness matrix (sparse for efficiency)
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(DOFs);  // Displacement increment vector

    const int dir = 0;
    // --- Start simulation timer --- //
    auto sim_start = std::chrono::high_resolution_clock::now();

    // Load stepping loop: gradually increase load from 0 to 1
    while (LF <= 1.0 + 1e-8) {
        std::cout << "\nLoad Factor: " << LF << std::endl;
        logger.writeLoadFactor(LF);
        
        // Apply prescribed boundary conditions (displacement or force) based on current load factor
        update_points(PL, LF, dx, Prescribed_Flag, F_prescribed, number_of_right_patches); 

        // Newton-Raphson iteration control variables
        int error_counter = 1;  // Iteration counter
        bool isNotAccurate = true;  // Convergence flag
        double normnull = 0.0;  // Initial residual norm for relative error calculation

        // Reset displacement increment for new load step
        dx.setZero();
        
        // Newton-Raphson iteration loop: solve nonlinear equilibrium at current load level
        while (isNotAccurate && error_counter <= max_try) {
            
            // Calculate internal forces and stiffness for all peridynamic bonds
            calculate_rk(PL, C1, C2, Delta, PD);

            // Assemble global residual vector
            assembly(PL, DOFs, R, K, "residual");

            // Optional: check if R is all zeros (quick sanity)
            if (R.array().isNaN().any() || R.array().isInf().any()) {
                std::cout << "ERROR: R contains NaN/Inf" << std::endl;
                break;
            }

            // Compute residual norm for convergence check
            double residual_norm = R.norm();
            double rel_norm;
            if (error_counter == 1) {
                // Store initial residual for relative error calculation
                normnull = std::max(residual_norm, 1e-10);
                std::cout << "Initial Residual Norm = " << residual_norm << std::endl;
            } else {
                // Check convergence based on relative or absolute tolerance
                rel_norm = residual_norm / normnull;
                std::cout << "Iter " << error_counter << ": Residual Norm = " << residual_norm << ", Relative = " << rel_norm << std::endl;
                if ((rel_norm - tol) < 1e-12 || (residual_norm - tol) < 1e-12) {
                    isNotAccurate = false;
                }
            }

            // Log convergence information
            logger.writeConvergence(error_counter, residual_norm, rel_norm);
            
            // Assemble global stiffness matrix
            assembly(PL, DOFs, R, K, "stiffness");

            std::cout << "K: rows=" << K.rows() << " cols=" << K.cols()
            << " nnz=" << K.nonZeros() << std::endl;

            // Optional basic diag sanity
            double minDiag = std::numeric_limits<double>::infinity();
            double maxDiag = 0.0;
            for (int k = 0; k < K.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K,k); it; ++it) {
                    if (it.row() == it.col()) {
                        double a = std::abs(it.value());
                        minDiag = std::min(minDiag, a);
                        maxDiag = std::max(maxDiag, a);
                    }
                }
            }
            std::cout << "K diag |min|=" << minDiag << " |max|=" << maxDiag << std::endl;
            // Solve linear system K * dx = -R using sparse LU decomposition
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(K);
            if(solver.info() != Eigen::Success) {
                std::cout << "Nonlinear solver failed to compute!" << std::endl;
                break;
            }
            dx = solver.solve(-R);  
            

            
            
            // Update point positions with calculated displacement increment
            update_points(PL, LF, dx, "Calculated", F_prescribed, number_of_right_patches);

            // Reset reaction force accumulators
            F_rec_patch.setZero();
            F_rec_rightpatch.setZero();
            
            // Accumulate reaction forces from boundary patches
            // Different calculation depending on whether force or displacement is prescribed
            for (int i = 0; i < PL.size(); i++) {
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Force")
                    F_rec_rightpatch += PL[i].F_ext;
                if (PL[i].Flag == "Right Patch" && Prescribed_Flag == "Displacement")
                    F_rec_rightpatch += PL[i].residual;
                if ((PL[i].Flag == "Patch"))
                    F_rec_patch += PL[i].residual;
            }

            // At final converged state of final load step, log detailed patch force data
            if (!isNotAccurate && LF >= 1.0 - 1e-12) {
                const int H = number_of_patches;

                // Collect residual forces from left patch nodes
                std::vector<Eigen::Vector3d> left_residuals;
                std::vector<double> left_residuals_comp;

                for (int i = 0; i < (int)PL.size(); ++i) {
                    if (PL[i].Flag == "Patch") {
                        left_residuals.push_back(PL[i].residual);
                        left_residuals_comp.push_back(PL[i].residual(dir)); // <-- scalar component
                    }
                }

                // Write patch forces to log file
                logger.writePatchForces(H, nn, left_residuals_comp, F_rec_rightpatch(dir));

                // Append force distribution data to CSV for post-processing
                bool file_exists = std::filesystem::exists("csv_files/force_by_position.csv");
                std::ofstream ofs("csv_files/force_by_position.csv", std::ios::app);
                if (!file_exists) {
                    ofs << "H,NN,X,Diff\n";
                }
                for (int k = 0; k < (int)left_residuals.size(); ++k) {
                    int Xpos = -(k+1);
                    ofs << H << "," << nn << "," << Xpos << "," << (F_rec_rightpatch - left_residuals[k]) << "\n";
                }
            }

            // Report successful convergence
            if(isNotAccurate == false) {
                std::cout << "Converged after " << error_counter << " iterations." << std::endl<< std::endl;
                logger.writeConverged(error_counter);
            }
            
            error_counter++;
        }
        
        // Increment load factor for next load step
        LF += load_step;   
    }
    
    // Display final reaction forces after all load steps completed
    std::cout<<"Applied / Reaction Force on the RIGHT PATCH is : "<< F_rec_rightpatch  <<std::endl;
    std::cout<<"Reaction Force on the PATCH is : "<< F_rec_patch <<std::endl;
    std::cout<<"Total Reaction force = Rightpatch - Patch = " << (F_rec_rightpatch - F_rec_patch)<< std::endl<< std::endl;
    logger.writeReactoinForce(LF,
                          F_rec_rightpatch(dir),
                          F_rec_patch(dir),
                          number_of_patches,
                          nn);
    
    // Optional: Output final state of all points (currently commented out)
    /*for (const auto& p : PL) {
        std::cout << "Point " << p.Nr << ": x = " << p.x << ",\t displacement = " << (p.x - p.X) << std::endl;
    }*/
    
    // --- End simulation timer --- //
    auto sim_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sim_duration = sim_end - sim_start;

    // --- End total timer --- //
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    // Display timing statistics
    std::cout << "\nSimulation time: " << sim_duration.count() << " seconds" << std::endl;
    std::cout << "Total program time: " << total_duration.count() << " seconds" << std::endl;
    logger.writeTiming(sim_duration.count(), total_duration.count());

    // Close logger and exit program
    logger.close();
    return 0;
}