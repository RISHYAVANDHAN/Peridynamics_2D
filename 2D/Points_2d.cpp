// Points.cpp
#include <set>
#include <iomanip>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <iostream>
#include <Eigen/Sparse>
#include "hyperdual.h"
#include "Points_2d.h"

// ============================================================================
// COMPUTE CORNERS: Generate corner points for 2D and 3D domains
// ============================================================================
std::vector<Eigen::Vector3d> Compute_Corners(int PD, double SiZe) {
    
    int length = std::pow(2, PD); // Length = Number of Corners (2^PD)
    std::vector<Eigen::Vector3d> Corners;
    Corners.resize(length);
    
    if (PD == 2) {
        // 2D corners: square with side length = SiZe, centered at origin
        Corners[0] = 0.5 * SiZe * Eigen::Vector3d(-1, -1, 0);  // Bottom-left
        Corners[1] = 0.5 * SiZe * Eigen::Vector3d( 1, -1, 0);  // Bottom-right
        Corners[2] = 0.5 * SiZe * Eigen::Vector3d( 1,  1, 0);  // Top-right
        Corners[3] = 0.5 * SiZe * Eigen::Vector3d(-1,  1, 0);  // Top-left
    } 
    else if (PD == 3) {
        // 3D corners: cube with side length = SiZe, centered at origin
        Corners[0] = 0.5 * SiZe * Eigen::Vector3d(-1, -1, -1); // Left-south-bottom
        Corners[1] = 0.5 * SiZe * Eigen::Vector3d( 1, -1, -1); // Right-south-bottom
        Corners[2] = 0.5 * SiZe * Eigen::Vector3d( 1,  1, -1); // Right-north-bottom
        Corners[3] = 0.5 * SiZe * Eigen::Vector3d(-1,  1, -1); // Left-north-bottom
        Corners[4] = 0.5 * SiZe * Eigen::Vector3d(-1, -1,  1); // Left-south-top
        Corners[5] = 0.5 * SiZe * Eigen::Vector3d( 1, -1,  1); // Right-south-top
        Corners[6] = 0.5 * SiZe * Eigen::Vector3d( 1,  1,  1); // Right-north-top
        Corners[7] = 0.5 * SiZe * Eigen::Vector3d(-1,  1,  1); // Left-north-top
    }
    
    return Corners;
}

// ============================================================================
// MESH: Generate nodal coordinates and element connectivity for 2D/3D domains
// ============================================================================
std::pair<std::vector<Eigen::Vector3d>, std::vector<std::vector<int>>>
Mesh(int PD, const std::vector<Eigen::Vector3d>& Corners, double L)
{
    std::vector<Eigen::Vector3d> NL;  // Node List - coordinates of all nodes
    std::vector<std::vector<int>> CNCT; // Connectivity - element-node relationships
    int Nx = 0, Ny = 0, Nz = 0; // Number of nodes in x, y, z directions
    
    // ----------------------------------------------------
    // Generate NL and CNCT for 2D
    // ----------------------------------------------------
    if (PD == 2)
    {
        // Extract corner points for 2D domain
        Eigen::Vector3d A = Corners[0]; // Bottom-left corner
        Eigen::Vector3d B = Corners[1]; // Bottom-right corner
        Eigen::Vector3d D = Corners[3]; // Top-left corner

        // Calculate number of nodes in each direction based on spacing L
        Nx = std::round((B.x() - A.x()) / L) + 1;
        Ny = std::round((D.y() - A.y()) / L) + 1;

        // Generate coordinate arrays in x and y directions
        std::vector<double> xx(Nx), yy(Ny);
        for (int i = 0; i < Nx; i++)
            xx[i] = A.x() + i * (B.x() - A.x()) / (Nx - 1);
        for (int j = 0; j < Ny; j++)
            yy[j] = A.y() + j * (D.y() - A.y()) / (Ny - 1);

        // Create nodal coordinates in a structured grid
        NL.resize(Nx * Ny);
        for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++)
                NL[j * Nx + i] = Eigen::Vector3d(xx[i], yy[j], 0.0);

        // ----------------------------------------------------
        // Generate CNCT for 2D quad elements (4 nodes per element)
        // ----------------------------------------------------
        CNCT.resize((Ny - 1) * (Nx - 1), std::vector<int>(4));
        for (int j = 0; j < Ny - 1; j++)
        {
            for (int i = 0; i < Nx - 1; i++)
            {
                int e = j * (Nx - 1) + i; // Element index
                // Define element connectivity in counter-clockwise order
                CNCT[e][0] = j * Nx + i;         // Bottom-left node
                CNCT[e][1] = j * Nx + i + 1;     // Bottom-right node
                CNCT[e][2] = (j + 1) * Nx + i + 1; // Top-right node
                CNCT[e][3] = (j + 1) * Nx + i;   // Top-left node
            }
        }
    }
    // ----------------------------------------------------
    // Generate NL and CNCT for 3D
    // ----------------------------------------------------
    else if (PD == 3)
    {
        // Extract corner points for 3D domain
        Eigen::Vector3d A = Corners[0]; // Left-south-bottom corner
        Eigen::Vector3d B = Corners[1]; // Right-south-bottom corner
        Eigen::Vector3d D = Corners[3]; // Left-north-bottom corner
        Eigen::Vector3d E = Corners[4]; // Left-south-top corner

        // Calculate number of nodes in each direction
        Nx = std::round((B.x() - A.x()) / L) + 1;
        Ny = std::round((D.y() - A.y()) / L) + 1;
        Nz = std::round((E.z() - A.z()) / L) + 1;

        // Generate coordinate arrays in x, y, z directions
        std::vector<double> xx(Nx), yy(Ny), zz(Nz);
        for (int i = 0; i < Nx; i++)
            xx[i] = A.x() + i * (B.x() - A.x()) / (Nx - 1);
        for (int j = 0; j < Ny; j++)
            yy[j] = A.y() + j * (D.y() - A.y()) / (Ny - 1);
        for (int k = 0; k < Nz; k++)
            zz[k] = A.z() + k * (E.z() - A.z()) / (Nz - 1);

        // Create nodal coordinates in a structured 3D grid
        NL.resize(Nx * Ny * Nz);
        for (int k = 0; k < Nz; k++)
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++)
                    NL[k * Ny * Nx + j * Nx + i] = Eigen::Vector3d(xx[i], yy[j], zz[k]);

        // ----------------------------------------------------
        // Generate CNCT for 3D brick elements (8 nodes per element)
        // ----------------------------------------------------
        CNCT.resize((Nz - 1) * (Ny - 1) * (Nx - 1), std::vector<int>(8));
        for (int k = 0; k < Nz - 1; k++)
        {
            for (int j = 0; j < Ny - 1; j++)
            {
                for (int i = 0; i < Nx - 1; i++)
                {
                    int e = k * (Ny - 1) * (Nx - 1) + j * (Nx - 1) + i; // Element index
                    // Define element connectivity (bottom face first, then top face)
                    CNCT[e][0] = k * Ny * Nx + j * Nx + i;         // Bottom-back-left
                    CNCT[e][1] = k * Ny * Nx + j * Nx + i + 1;     // Bottom-back-right
                    CNCT[e][2] = k * Ny * Nx + (j + 1) * Nx + i + 1; // Bottom-front-right
                    CNCT[e][3] = k * Ny * Nx + (j + 1) * Nx + i;   // Bottom-front-left
                    CNCT[e][4] = (k + 1) * Ny * Nx + j * Nx + i;   // Top-back-left
                    CNCT[e][5] = (k + 1) * Ny * Nx + j * Nx + i + 1; // Top-back-right
                    CNCT[e][6] = (k + 1) * Ny * Nx + (j + 1) * Nx + i + 1; // Top-front-right
                    CNCT[e][7] = (k + 1) * Ny * Nx + (j + 1) * Nx + i; // Top-front-left
                }
            }
        }
    }

    return { NL, CNCT };
}

// ============================================================================
// PATCH NODE: Check if a node is outside the original domain boundaries
// ============================================================================
bool PatchNode(const Eigen::VectorXd& node, const std::vector<Eigen::Vector3d>& Corners) {
    bool out = false;    
    double tol = 1e-4; // Tolerance for boundary check
    int PD = node.size();
    
    if (PD == 2) {
        // Check if node is outside the original 2D domain boundaries
        if ((node(0) - Corners[0].x()) < -tol || (node(1) - Corners[0].y()) < -tol ||
            (node(0) - Corners[1].x()) > tol || (node(1) - Corners[3].y()) > tol) {
            out = true;
        }
    }
    else if (PD == 3) {
        // Check if node is outside the original 3D domain boundaries
        if ((node(0) - Corners[0].x()) < -tol || (node(1) - Corners[0].y()) < -tol ||
            (node(2) - Corners[0].z()) < -tol || (node(0) - Corners[1].x()) > tol ||
            (node(1) - Corners[3].y()) > tol || (node(2) - Corners[4].z()) > tol) {
            out = true;
        }
    }
    
    return out;
}

// ============================================================================
// PATCH: Generate patch nodes around the original domain
// ============================================================================
std::vector<Eigen::Vector3d> Patch(int PD, const std::vector<Eigen::Vector3d>& Corners, 
                                  double L, double Delta, int patch, int right_patch) {
    // Calculate patch extensions for left and right sides
    double l = patch * L;      // Left patch extension
    double r = right_patch * L; // Right patch extension

    int length = std::pow(2, PD);
    std::vector<Eigen::Vector3d> Corners_mod;
    Corners_mod.resize(length);

    if (PD == 2) {
        // Extend 2D corners with left (l) on left side and right (r) on right side
        Corners_mod[0] = Corners[0] + l * Eigen::Vector3d(-1, -1, 0);  // Left-south corner extended
        Corners_mod[1] = Corners[1] + r * Eigen::Vector3d( 1, -1, 0);  // Right-south corner extended
        Corners_mod[2] = Corners[2] + r * Eigen::Vector3d( 1,  1, 0);  // Right-north corner extended
        Corners_mod[3] = Corners[3] + l * Eigen::Vector3d(-1,  1, 0);  // Left-north corner extended
    }
    else if (PD == 3) {
        // Extend 3D corners with left (l) and right (r) patches
        Corners_mod[0] = Corners[0] + l * Eigen::Vector3d(-1, -1, -1);
        Corners_mod[1] = Corners[1] + r * Eigen::Vector3d( 1, -1, -1);
        Corners_mod[2] = Corners[2] + r * Eigen::Vector3d( 1,  1, -1);
        Corners_mod[3] = Corners[3] + l * Eigen::Vector3d(-1,  1, -1);
        Corners_mod[4] = Corners[4] + l * Eigen::Vector3d(-1, -1,  1);
        Corners_mod[5] = Corners[5] + r * Eigen::Vector3d( 1, -1,  1);
        Corners_mod[6] = Corners[6] + r * Eigen::Vector3d( 1,  1,  1);
        Corners_mod[7] = Corners[7] + l * Eigen::Vector3d(-1,  1,  1);
    }

    // Generate a temporary mesh over the extended domain
    auto [NLtmp, CNect_tmp] = Mesh(PD, Corners_mod, L);

    int NoNs = NLtmp.size();
    std::vector<Eigen::Vector3d> NL;

    // Filter the nodes: only keep those that are *outside* the original domain
    for (int i = 0; i < NoNs; i++) {
        Eigen::VectorXd node(PD);
        for (int d = 0; d < PD; d++) {
            node(d) = NLtmp[i](d);
        }

        if (PatchNode(node, Corners)) {
            NL.emplace_back(NLtmp[i]);
        }
    }

    return NL;
}

// ============================================================================
// TOPOLOGY: Create point list and element list based on topology flag
// ============================================================================
std::pair<std::vector<Point>, std::vector<Element>> 
Topology(int PD, const std::vector<Eigen::Vector3d>& NL, const std::vector<std::vector<int>>& CNCT, 
         double L, double Delta, const std::vector<double>& Bvals, const std::string& TOPflag) {
    int NoNs = NL.size();  // Number of nodes
    int NoP = 0;           // Point number counter
    std::vector<int> NPL(NoNs, 0);  // Mapping from original node index to point list index
    std::vector<Point> PL; // Point list
    std::vector<Element> EL; // Element list
    
    double tol = 1e-3; // Tolerance for geometric checks
    
    // ========== Create Point List based on TOPflag ==========
    if (TOPflag == "FULL") {
        // Keep all nodes in the domain
        for (int n = 0; n < NoNs; n++) {
            NoP = NoP + 1;
            Point newPoint(NoP, NL[n]); // Create point with initial position
            newPoint.NNr = n;           // Store original node index
            newPoint.PD = PD;           // Set problem dimension
            newPoint.F_ext = Eigen::VectorXd::Zero(PD);    // Set size to 2 or 3 and zero it
            newPoint.residual = Eigen::VectorXd::Zero(PD); // Set size to 2 or 3 and zero it
            // -----------------------
            PL.emplace_back(newPoint);
            NPL[n] = NoP;               // Map node index to point index
        }
    }
    else if (TOPflag == "PART") {
        // Keep nodes outside a specified sphere
        if (PD == 2) {
            double R = Bvals[0];  // Sphere radius
            double Xo = Bvals[1]; // Sphere center x-coordinate
            double Yo = Bvals[2]; // Sphere center y-coordinate
            
            for (int n = 0; n < NoNs; n++) {
                double dx = NL[n].x() - Xo;
                double dy = NL[n].y() - Yo;
                double distSq = dx*dx + dy*dy;
                
                // Keep node if it's outside or on the sphere boundary
                if ((distSq - R*R) > -tol) {
                    NoP = NoP + 1;
                    Point newPoint(NoP, NL[n]);
                    newPoint.NNr = n;
                    newPoint.PD = PD;
                    PL.emplace_back(newPoint);
                    NPL[n] = NoP;
                }
            }
        }
        else if (PD == 3) {
            double R = Bvals[0];  // Sphere radius
            double Xo = Bvals[1]; // Sphere center x-coordinate
            double Yo = Bvals[2]; // Sphere center y-coordinate
            double Zo = Bvals[3]; // Sphere center z-coordinate
            
            for (int n = 0; n < NoNs; n++) {
                double dx = NL[n].x() - Xo;
                double dy = NL[n].y() - Yo;
                double dz = NL[n].z() - Zo;
                double distSq = dx*dx + dy*dy + dz*dz;
                
                // Keep node if it's outside or on the sphere boundary
                if ((distSq - R*R) > -tol) {
                    NoP = NoP + 1;
                    Point newPoint(NoP, NL[n]);
                    newPoint.NNr = n;
                    newPoint.PD = PD;
                    PL.emplace_back(newPoint);
                    NPL[n] = NoP;
                }
            }
        }
    }
    else if (TOPflag == "CUT") {
        // Keep nodes on boundary of a box
        double v = 0;  // Additional layer parameter (if non-zero, will have additional layers)
        
        if (PD == 2) {
            double R = Bvals[0] - v; // Box half-size
            double Xo = Bvals[1];    // Box center x-coordinate
            double Yo = Bvals[2];    // Box center y-coordinate
            
            for (int n = 0; n < NoNs; n++) {
                double x = NL[n].x();
                double y = NL[n].y();
                
                // Keep node if it's on the box boundary
                if ((x - (Xo + R)) > -tol || (x - (Xo - R)) < tol || 
                    (y - (Yo + R)) > -tol || (y - (Yo - R)) < tol) {
                    NoP = NoP + 1;
                    Point newPoint(NoP, NL[n]);
                    newPoint.NNr = n;
                    newPoint.PD = PD;
                    PL.emplace_back(newPoint);
                    NPL[n] = NoP;
                }
            }
        }
        else if (PD == 3) {
            double R = Bvals[0] - v; // Box half-size
            double Xo = Bvals[1];    // Box center x-coordinate
            double Yo = Bvals[2];    // Box center y-coordinate
            double Zo = Bvals[3];    // Box center z-coordinate
            
            for (int n = 0; n < NoNs; n++) {
                double x = NL[n].x();
                double y = NL[n].y();
                double z = NL[n].z();
                
                // Keep node if it's on the box boundary
                if ((x - (Xo + R)) > -tol || (x - (Xo - R)) < tol || 
                    (y - (Yo + R)) > -tol || (y - (Yo - R)) < tol ||
                    (z - (Zo + R)) > -tol || (z - (Zo - R)) < tol) {
                    NoP = NoP + 1;
                    Point newPoint(NoP, NL[n]);
                    newPoint.NNr = n;
                    newPoint.PD = PD;
                    PL.emplace_back(newPoint);
                    NPL[n] = NoP;
                }
            }
        }
    }
    
    // ========== Create Element List if needed ==========
    int NoEs = CNCT.size();
    int NPE = (NoEs > 0) ? CNCT[0].size() : 0;  // Nodes per element (4 for 2D, 8 for 3D)
    
    if (TOPflag == "FULL") {
        // Create all elements
        for (int e = 0; e < NoEs; e++) {
            std::vector<int> NdL = CNCT[e]; // Original node indices for this element
            std::vector<int> PNdL(NPE);     // Point indices for this element
            Eigen::MatrixXd XX(PD, NPE);    // Element node coordinates
            
            for (int i = 0; i < NPE; i++) {
                XX.col(i) = NL[NdL[i]].head(PD); // Get coordinates (x,y) or (x,y,z)
                PNdL[i] = NPL[NdL[i]];           // Map to point index
            }
            
            Element newElem(e + 1, PNdL, XX); // Create element
            EL.emplace_back(newElem);
        }
    }
    else if (TOPflag == "PART") {
        // Create elements only if all nodes are outside the sphere
        if (PD == 2) {
            double R = Bvals[0];
            double Xo = Bvals[1];
            double Yo = Bvals[2];
            
            for (int e = 0; e < NoEs; e++) {
                std::vector<int> NdL = CNCT[e];
                std::vector<int> PNdL(NPE);
                Eigen::MatrixXd XX(PD, NPE);
                
                for (int i = 0; i < NPE; i++) {
                    XX.col(i) = NL[NdL[i]].head(PD);
                    PNdL[i] = NPL[NdL[i]];
                }
                
                // Check if all 4 nodes are outside sphere
                bool allOutside = true;
                for (int i = 0; i < 4; i++) {
                    double dx = XX(0, i) - Xo;
                    double dy = XX(1, i) - Yo;
                    if ((dx*dx + dy*dy - R*R) <= -tol) {
                        allOutside = false;
                        break;
                    }
                }
                
                if (allOutside) {
                    Element newElem(EL.size() + 1, PNdL, XX);
                    EL.emplace_back(newElem);
                }
            }
        }
        else if (PD == 3) {
            double R = Bvals[0];
            double Xo = Bvals[1];
            double Yo = Bvals[2];
            double Zo = Bvals[3];
            
            for (int e = 0; e < NoEs; e++) {
                std::vector<int> NdL = CNCT[e];
                std::vector<int> PNdL(NPE);
                Eigen::MatrixXd XX(PD, NPE);
                
                for (int i = 0; i < NPE; i++) {
                    XX.col(i) = NL[NdL[i]].head(PD);
                    PNdL[i] = NPL[NdL[i]];
                }
                
                // Check if all 8 nodes are outside sphere
                bool allOutside = true;
                for (int i = 0; i < 8; i++) {
                    double dx = XX(0, i) - Xo;
                    double dy = XX(1, i) - Yo;
                    double dz = XX(2, i) - Zo;
                    if ((dx*dx + dy*dy + dz*dz - R*R) <= -tol) {
                        allOutside = false;
                        break;
                    }
                }
                
                if (allOutside) {
                    Element newElem(EL.size() + 1, PNdL, XX);
                    EL.emplace_back(newElem);
                }
            }
        }
    }
    else if (TOPflag == "CUT") {
        // Create elements only if all nodes are on/outside box boundary
        double v = 0;
        
        if (PD == 2) {
            double R = Bvals[0] - v;
            double Xo = Bvals[1];
            double Yo = Bvals[2];
            
            for (int e = 0; e < NoEs; e++) {
                std::vector<int> NdL = CNCT[e];
                std::vector<int> PNdL(NPE);
                Eigen::MatrixXd XX(PD, NPE);
                
                for (int i = 0; i < NPE; i++) {
                    XX.col(i) = NL[NdL[i]].head(PD);
                    PNdL[i] = NPL[NdL[i]];
                }
                
                // Check if all 4 nodes are on boundary
                bool allOnBoundary = true;
                for (int i = 0; i < 4; i++) {
                    double x = XX(0, i);
                    double y = XX(1, i);
                    if (!((x - (Xo + R)) > -tol || (x - (Xo - R)) < tol || 
                          (y - (Yo + R)) > -tol || (y - (Yo - R)) < tol)) {
                        allOnBoundary = false;
                        break;
                    }
                }
                
                if (allOnBoundary) {
                    Element newElem(EL.size() + 1, PNdL, XX);
                    EL.emplace_back(newElem);
                }
            }
        }
        else if (PD == 3) {
            double R = Bvals[0] - v;
            double Xo = Bvals[1];
            double Yo = Bvals[2];
            double Zo = Bvals[3];
            
            for (int e = 0; e < NoEs; e++) {
                std::vector<int> NdL = CNCT[e];
                std::vector<int> PNdL(NPE);
                Eigen::MatrixXd XX(PD, NPE);
                
                for (int i = 0; i < NPE; i++) {
                    XX.col(i) = NL[NdL[i]].head(PD);
                    PNdL[i] = NPL[NdL[i]];
                }
                
                // Check if all 8 nodes are on boundary
                bool allOnBoundary = true;
                for (int i = 0; i < 8; i++) {
                    double x = XX(0, i);
                    double y = XX(1, i);
                    double z = XX(2, i);
                    if (!((x - (Xo + R)) > -tol || (x - (Xo - R)) < tol || 
                          (y - (Yo + R)) > -tol || (y - (Yo - R)) < tol ||
                          (z - (Zo + R)) > -tol || (z - (Zo - R)) < tol)) {
                        allOnBoundary = false;
                        break;
                    }
                }
                
                if (allOnBoundary) {
                    Element newElem(EL.size() + 1, PNdL, XX);
                    EL.emplace_back(newElem);
                }
            }
        }
    }
    
    return {PL, EL};
}

// ============================================================================
// ASSIGN NGBRS: Find neighbors for each point within horizon distance Delta
// ============================================================================
std::vector<Point> AssignNgbrs(int PD, std::vector<Point> PL, double L, double Delta) {
    int NoPs = PL.size();    // Number of points
    double tol = 1e-8;       // Tolerance for geometric checks
    
    // Calculate NmaxNgbr (Maximum number of neighbors in a full horizon)
    int Del_by_L = static_cast<int>(floor(Delta / L));
    int NmaxNgbr = 0;
    
    // Count maximum possible neighbors in perfect grid within horizon
    if (PD == 2) {
        for (int i = -Del_by_L; i <= Del_by_L; i++) {
            for (int j = -Del_by_L; j <= Del_by_L; j++) {
                if ((sqrt(i*i + j*j) * L < Delta) && (i != 0 || j != 0)) {
                    NmaxNgbr++;
                }
            }
        }
    }
    else if (PD == 3) {
        for (int i = -Del_by_L; i <= Del_by_L; i++) {
            for (int j = -Del_by_L; j <= Del_by_L; j++) {
                for (int k = -Del_by_L; k <= Del_by_L; k++) {
                    if ((sqrt(i*i + j*j + k*k) * L < Delta) && (i != 0 || j != 0 || k != 0)) {
                        NmaxNgbr = NmaxNgbr + 1;
                    }
                }
            }
        }
    }
    
    // For each point, find all neighbors within distance Delta
    for (int p = 0; p < NoPs; p++) {
        std::vector<int> neighbors;              // Neighbor point indices
        std::vector<Eigen::Vector3d> neighborsX; // Neighbor initial positions
        std::vector<Eigen::Vector3d> neighborsx; // Neighbor current positions
        
        // Find all neighbors within distance Delta
        for (int q = 0; q < NoPs; q++) {
            if (q != p) { // Exclude self
                double distance = (PL[p].X - PL[q].X).norm(); // Euclidean distance
                if (distance < Delta) {
                    neighbors.emplace_back(q);
                    neighborsX.emplace_back(PL[q].X);
                    neighborsx.emplace_back(PL[q].x);
                }
            }
        }
        
        int NNgbr = neighbors.size(); // Number of neighbors found
        
        // Calculate NInII (number of non-colinear pairs)
        int NInII = 0;
        if (PD == 2) {
            for (int i = 0; i < NNgbr; i++) {
                for (int j = 0; j < NNgbr; j++) {
                    if (j != i) {
                        Eigen::Vector3d XiI = neighborsX[i] - PL[p].X;  // Vector to neighbor i
                        Eigen::Vector3d XiII = neighborsX[j] - PL[p].X; // Vector to neighbor j
                        Eigen::Vector3d A = XiI.cross(XiII); // Cross product (area)
                        // Check if vectors are non-colinear and within horizon
                        if (A.norm() > tol && (XiI - XiII).norm() < Delta) {
                            NInII++;
                        }
                    }
                }
            }
        }
        else if (PD == 3) {
            for (int i = 0; i < NNgbr; i++) {
                for (int j = 0; j < NNgbr; j++) {
                    if (j != i) {
                        Eigen::Vector3d XiI = neighborsX[i] - PL[p].X;
                        Eigen::Vector3d XiII = neighborsX[j] - PL[p].X;
                        Eigen::Vector3d A = XiI.cross(XiII);
                        if (A.norm() > tol && (XiI - XiII).norm() < Delta) {
                            NInII = NInII + 1;
                        }
                    }
                }
            }
        }
        
        // Calculate NInIInIII (number of non-coplanar triplets) - 3D only
        int NInIInIII = 0;
        if (PD == 3) {
            for (int i = 0; i < NNgbr; i++) {
                for (int j = 0; j < NNgbr; j++) {
                    if (j != i) {
                        for (int k = 0; k < NNgbr; k++) {
                            if ((k != i) && (k != j)) {
                                Eigen::Vector3d XiI = neighborsX[i] - PL[p].X;
                                Eigen::Vector3d XiII = neighborsX[j] - PL[p].X;
                                Eigen::Vector3d XiIII = neighborsX[k] - PL[p].X;
                                double V = XiI.dot(XiII.cross(XiIII)); // Scalar triple product (volume)
                                // Check if vectors are non-coplanar and within horizon
                                if (std::abs(V) > tol && 
                                    (XiI - XiII).norm() < Delta && 
                                    (XiI - XiIII).norm() < Delta && 
                                    (XiII - XiIII).norm() < Delta) {
                                    NInIInIII = NInIInIII + 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate area/volume correction factor AV for volume normalization
        double AV = 0.0;
        if (PD == 2) {
            double Amax = M_PI * Delta * Delta; // Maximum area (circle)
            AV = static_cast<double>(NNgbr + 1) / (NmaxNgbr + 1) * Amax;
        }
        else if (PD == 3) {
            double Vmax = (4.0/3.0) * M_PI * Delta * Delta * Delta; // Maximum volume (sphere)
            AV = static_cast<double>(NNgbr + 1) / (NmaxNgbr + 1) * Vmax;
        }
        
        // Store all calculated neighborhood information in the Point object
        PL[p].neighbors = neighbors;
        PL[p].neighborsX_vec = neighborsX;
        PL[p].neighborsx_vec = neighborsx;
        PL[p].NI = NNgbr;
        PL[p].NInII = NInII;
        PL[p].NInIInIII = NInIInIII;
        PL[p].AV = AV;
    }
    
    return PL;
}

// ============================================================================
// ASSIGN VOLS: Assign volumes to points based on their position in the domain
// ============================================================================
std::vector<Point> AssignVols(const std::vector<Eigen::Vector3d>& Corners, std::vector<Point> PL, 
                             double L, const std::vector<double>& Bvals, const std::string& TOPflag) {
    int NoPs = PL.size();
    int PD = PL[0].PD;
    double tol = 1e-4; // Tolerance for boundary checks

    if (TOPflag == "CUT") {
        // For CUT topology: complex volume assignment near boundaries and cut regions
        if (PD == 2) {
            Eigen::Vector3d A = Corners[0]; // Bottom-left corner
            Eigen::Vector3d B = Corners[1]; // Bottom-right corner
            Eigen::Vector3d C = Corners[2]; // Top-right corner
            Eigen::Vector3d D = Corners[3]; // Top-left corner

            double R = Bvals[0];  // Cut region parameter
            double Xo = Bvals[1]; // Cut center x-coordinate
            double Yo = Bvals[2]; // Cut center y-coordinate

            for (int p = 0; p < NoPs; p++) {
                Eigen::Vector3d X = PL[p].X;
                double alpha = 0.0; // Volume fraction

                // Determine volume fraction based on point position
                if ((X(0)-A(0)) < (-tol) || (X(0)-B(0)) > tol || 
                    (X(1)-A(1)) < (-tol) || (X(1)-D(1)) > tol) {
                    // Point is outside the primary domain (Patch node)
                    alpha = 0;
                } else if (std::abs(X(0)-A(0)) < tol || std::abs(X(0)-B(0)) < tol || 
                          std::abs(X(1)-A(1)) < tol || std::abs(X(1)-D(1)) < tol) {
                    // Point is on the domain boundary
                    if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol) ||
                        (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol) ||
                        (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol) ||
                        (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol)) {
                        // Corner point: 1/4 volume
                        alpha = 1.0/4.0;
                    } else {
                        // Edge point: 1/2 volume
                        alpha = 1.0/2.0;
                    }
                } else {
                    // Point is in the domain interior
                    if ((X(0) - (Xo+R)) > tol || (X(0) - (Xo-R)) < -tol || 
                        (X(1) - (Yo+R)) > tol || (X(1) - (Yo-R)) < -tol) {
                        // Point is outside cut region: full volume
                        alpha = 1;
                    } else if (std::abs(X(0) - (Xo+R)) < tol || std::abs(X(0) - (Xo-R)) < tol || 
                              std::abs(X(1) - (Yo+R)) < tol || std::abs(X(1) - (Yo-R)) < tol) {
                        // Point is on cut region boundary
                        if ((std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo-R)) < tol) ||
                            (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo+R)) < tol) ||
                            (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo-R)) < tol) ||
                            (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo+R)) < tol)) {
                            // Cut region corner: 3/4 volume
                            alpha = 3.0/4.0;
                        } else {
                            // Cut region edge: 1/2 volume
                            alpha = 1.0/2.0;
                        }
                    } else {
                        // Point is inside cut region: zero volume
                        alpha = 0;
                    }
                }

                double V = alpha * L * L; // 2D volume (area)
                PL[p].Vol = V;
            }
        }
        else if (PD == 3) {
            // 3D volume assignment for CUT topology
            Eigen::Vector3d A = Corners[0]; // Left-south-bottom
            Eigen::Vector3d B = Corners[1]; // Right-south-bottom
            Eigen::Vector3d C = Corners[2]; // Right-north-bottom
            Eigen::Vector3d D = Corners[3]; // Left-north-bottom
            Eigen::Vector3d E = Corners[4]; // Left-south-top
            Eigen::Vector3d F = Corners[5]; // Right-south-top
            Eigen::Vector3d G = Corners[6]; // Right-north-top
            Eigen::Vector3d H = Corners[7]; // Left-north-top

            double R = Bvals[0];  // Cut region parameter
            double Xo = Bvals[1]; // Cut center x-coordinate
            double Yo = Bvals[2]; // Cut center y-coordinate
            double Zo = Bvals[3]; // Cut center z-coordinate

            for (int p = 0; p < NoPs; p++) {
                Eigen::Vector3d X = PL[p].X;
                double alpha = 0.0;

                // Check if point is outside the entire 3D domain
                if ((X(0)-A(0)) < (-tol) || (X(0)-B(0)) > tol || 
                    (X(1)-A(1)) < (-tol) || (X(1)-D(1)) > tol || 
                    (X(2)-A(2)) < (-tol) || (X(2)-E(2)) > tol) {
                    alpha = 0;
                }
                // Check if point is on the domain boundary
                else if (std::abs(X(0)-A(0)) < tol || std::abs(X(0)-B(0)) < tol || 
                        std::abs(X(1)-A(1)) < tol || std::abs(X(1)-D(1)) < tol || 
                        std::abs(X(2)-A(2)) < tol || std::abs(X(2)-E(2)) < tol) {
                    
                    // Check for corner points (8 corners)
                    if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol && std::abs(X(2)-A(2)) < tol) ||
                        (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol && std::abs(X(2)-B(2)) < tol) ||
                        (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol && std::abs(X(2)-C(2)) < tol) ||
                        (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol && std::abs(X(2)-D(2)) < tol) ||
                        (std::abs(X(0)-E(0)) < tol && std::abs(X(1)-E(1)) < tol && std::abs(X(2)-E(2)) < tol) ||
                        (std::abs(X(0)-F(0)) < tol && std::abs(X(1)-F(1)) < tol && std::abs(X(2)-F(2)) < tol) ||
                        (std::abs(X(0)-G(0)) < tol && std::abs(X(1)-G(1)) < tol && std::abs(X(2)-G(2)) < tol) ||
                        (std::abs(X(0)-H(0)) < tol && std::abs(X(1)-H(1)) < tol && std::abs(X(2)-H(2)) < tol)) {
                        // Corner point: 1/8 volume
                        alpha = 1.0/8.0;
                    }
                    // Check for edge points (12 edges)
                    else if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol) ||
                            (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol) ||
                            (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol) ||
                            (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol) ||
                            (std::abs(X(0)-A(0)) < tol && std::abs(X(2)-A(2)) < tol) ||
                            (std::abs(X(0)-B(0)) < tol && std::abs(X(2)-B(2)) < tol) ||
                            (std::abs(X(0)-E(0)) < tol && std::abs(X(2)-E(2)) < tol) ||
                            (std::abs(X(0)-F(0)) < tol && std::abs(X(2)-F(2)) < tol) ||
                            (std::abs(X(1)-A(1)) < tol && std::abs(X(2)-A(2)) < tol) ||
                            (std::abs(X(1)-D(1)) < tol && std::abs(X(2)-D(2)) < tol) ||
                            (std::abs(X(1)-E(1)) < tol && std::abs(X(2)-E(2)) < tol) ||
                            (std::abs(X(1)-H(1)) < tol && std::abs(X(2)-H(2)) < tol)) {
                        // Edge point: 1/4 volume
                        alpha = 1.0/4.0;
                    }
                    else {
                        // Face point: 1/2 volume
                        alpha = 1.0/2.0;
                    }
                }
                else {
                    // Point is inside the domain interior
                    if ((X(0) - (Xo+R)) > tol || (X(0) - (Xo-R)) < -tol || 
                        (X(1) - (Yo+R)) > tol || (X(1) - (Yo-R)) < -tol || 
                        (X(2) - (Zo+R)) > tol || (X(2) - (Zo-R)) < -tol) {
                        // Point is outside cut region: full volume
                        alpha = 1;
                    }
                    else if (std::abs(X(0) - (Xo+R)) < tol || std::abs(X(0) - (Xo-R)) < tol || 
                            std::abs(X(1) - (Yo+R)) < tol || std::abs(X(1) - (Yo-R)) < tol || 
                            std::abs(X(2) - (Zo+R)) < tol || std::abs(X(2) - (Zo-R)) < tol) {
                        
                        // Check for cut region corners (8 corners)
                        if ((std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                            (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                            (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                            (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                            (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                            (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                            (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                            (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo+R)) < tol)) {
                            // Cut region corner: 7/8 volume
                            alpha = 7.0/8.0;
                        }
                        // Check for cut region edges (12 edges)
                        else if ((std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo-R)) < tol) ||
                                (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo-R)) < tol) ||
                                (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(1) - (Yo+R)) < tol) ||
                                (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(1) - (Yo+R)) < tol) ||
                                (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                                (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                                (std::abs(X(0) - (Xo-R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                                (std::abs(X(0) - (Xo+R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                                (std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                                (std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo-R)) < tol) ||
                                (std::abs(X(1) - (Yo-R)) < tol && std::abs(X(2) - (Zo+R)) < tol) ||
                                (std::abs(X(1) - (Yo+R)) < tol && std::abs(X(2) - (Zo+R)) < tol)) {
                            // Cut region edge: 3/4 volume
                            alpha = 3.0/4.0;
                        }
                        else {
                            // Cut region face: 1/2 volume
                            alpha = 1.0/2.0;
                        }
                    }
                    else {
                        // Point is inside cut region: zero volume
                        alpha = 0;
                    }
                }

                double V = alpha * L * L * L; // 3D volume
                PL[p].Vol = V;
            }
        }
    } else if (TOPflag == "FULL") {
        // For FULL topology: simpler volume assignment based on domain boundaries
        if (PD == 2) {
            Eigen::Vector3d A = Corners[0]; // Bottom-left
            Eigen::Vector3d B = Corners[1]; // Bottom-right
            Eigen::Vector3d C = Corners[2]; // Top-right
            Eigen::Vector3d D = Corners[3]; // Top-left

            for (int p = 0; p < NoPs; p++) {
                Eigen::Vector3d X = PL[p].X;
                double alpha = 0.0;

                if ((X(0)-A(0)) < (-tol) || (X(0)-B(0)) > tol || 
                    (X(1)-A(1)) < (-tol) || (X(1)-D(1)) > tol) {
                    // Outside domain
                    alpha = 0;
                } else if (std::abs(X(0)-A(0)) < tol || std::abs(X(0)-B(0)) < tol || 
                          std::abs(X(1)-A(1)) < tol || std::abs(X(1)-D(1)) < tol) {
                    // On boundary
                    if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol) ||
                        (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol) ||
                        (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol) ||
                        (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol)) {
                        // Corner: 1/4 volume
                        alpha = 1.0/4.0;
                    } else {
                        // Edge: 1/2 volume
                        alpha = 1.0/2.0;
                    }
                } else {
                    // Interior: full volume
                    alpha = 1;
                }

                double V = alpha * L * L;
                PL[p].Vol = V;
            }
        }
        else if (PD == 3) {
            // 3D volume assignment for FULL topology
            Eigen::Vector3d A = Corners[0]; // Left-south-bottom
            Eigen::Vector3d B = Corners[1]; // Right-south-bottom
            Eigen::Vector3d C = Corners[2]; // Right-north-bottom
            Eigen::Vector3d D = Corners[3]; // Left-north-bottom
            Eigen::Vector3d E = Corners[4]; // Left-south-top
            Eigen::Vector3d F = Corners[5]; // Right-south-top
            Eigen::Vector3d G = Corners[6]; // Right-north-top
            Eigen::Vector3d H = Corners[7]; // Left-north-top

            for (int p = 0; p < NoPs; p++) {
                Eigen::Vector3d X = PL[p].X;
                double alpha = 0.0;

                // Check if point is outside the entire 3D domain
                if ((X(0)-A(0)) < (-tol) || (X(0)-B(0)) > tol || 
                    (X(1)-A(1)) < (-tol) || (X(1)-D(1)) > tol || 
                    (X(2)-A(2)) < (-tol) || (X(2)-E(2)) > tol) {
                    alpha = 0;
                }
                // Check if point is on the domain boundary
                else if (std::abs(X(0)-A(0)) < tol || std::abs(X(0)-B(0)) < tol || 
                        std::abs(X(1)-A(1)) < tol || std::abs(X(1)-D(1)) < tol || 
                        std::abs(X(2)-A(2)) < tol || std::abs(X(2)-E(2)) < tol) {
                    
                    // Check for corner points (8 corners)
                    if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol && std::abs(X(2)-A(2)) < tol) ||
                        (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol && std::abs(X(2)-B(2)) < tol) ||
                        (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol && std::abs(X(2)-C(2)) < tol) ||
                        (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol && std::abs(X(2)-D(2)) < tol) ||
                        (std::abs(X(0)-E(0)) < tol && std::abs(X(1)-E(1)) < tol && std::abs(X(2)-E(2)) < tol) ||
                        (std::abs(X(0)-F(0)) < tol && std::abs(X(1)-F(1)) < tol && std::abs(X(2)-F(2)) < tol) ||
                        (std::abs(X(0)-G(0)) < tol && std::abs(X(1)-G(1)) < tol && std::abs(X(2)-G(2)) < tol) ||
                        (std::abs(X(0)-H(0)) < tol && std::abs(X(1)-H(1)) < tol && std::abs(X(2)-H(2)) < tol)) {
                        // Corner point: 1/8 volume
                        alpha = 1.0/8.0;
                    }
                    // Check for edge points (12 edges)
                    else if ((std::abs(X(0)-A(0)) < tol && std::abs(X(1)-A(1)) < tol) ||
                            (std::abs(X(0)-B(0)) < tol && std::abs(X(1)-B(1)) < tol) ||
                            (std::abs(X(0)-C(0)) < tol && std::abs(X(1)-C(1)) < tol) ||
                            (std::abs(X(0)-D(0)) < tol && std::abs(X(1)-D(1)) < tol) ||
                            (std::abs(X(0)-A(0)) < tol && std::abs(X(2)-A(2)) < tol) ||
                            (std::abs(X(0)-B(0)) < tol && std::abs(X(2)-B(2)) < tol) ||
                            (std::abs(X(0)-E(0)) < tol && std::abs(X(2)-E(2)) < tol) ||
                            (std::abs(X(0)-F(0)) < tol && std::abs(X(2)-F(2)) < tol) ||
                            (std::abs(X(1)-A(1)) < tol && std::abs(X(2)-A(2)) < tol) ||
                            (std::abs(X(1)-D(1)) < tol && std::abs(X(2)-D(2)) < tol) ||
                            (std::abs(X(1)-E(1)) < tol && std::abs(X(2)-E(2)) < tol) ||
                            (std::abs(X(1)-H(1)) < tol && std::abs(X(2)-H(2)) < tol)) {
                        // Edge point: 1/4 volume
                        alpha = 1.0/4.0;
                    }
                    else {
                        // Face point: 1/2 volume
                        alpha = 1.0/2.0;
                    }
                }
                else {
                    // Interior point: full volume
                    alpha = 1;
                }

                double V = alpha * L * L * L; // 3D volume
                PL[p].Vol = V;
            }
        }
    }
    
    return PL;
}

// ============================================================================
// SET MATERIAL: Assign material properties to points or elements
// ============================================================================

// For Points
std::vector<Point> SetMaterial(const std::vector<Point>& inp, double L, double Delta, 
                              const std::vector<double>& MatPars, const std::string& MatLaw) {
    std::vector<Point> PL = inp;
    int NoPs = PL.size();
    
    for (int p = 0; p < NoPs; p++) {
        int mat = 1; // Material ID (can be extended for multiple materials)
        PL[p].L = L;
        PL[p].Delta = Delta;
        PL[p].Mat = mat;
        PL[p].MatPars = MatPars;
        PL[p].MatLaw = MatLaw;
    }
    
    return PL;
}

// For Elements (overloaded function)
std::vector<Element> SetMaterial(const std::vector<Element>& inp, const std::vector<double>& MatPars, 
                                const std::string& MatLaw) {
    std::vector<Element> EL = inp;
    int NoEs = EL.size();
    
    for (int e = 0; e < NoEs; e++) {
        int mat = 1;
        EL[e].Mat = mat;
        EL[e].MatPars = MatPars;
        EL[e].MatLaw = MatLaw;
    }
    
    return EL;
}

// ============================================================================
// COMPUTE FF: Compute deformation gradient tensor for different deformation modes
// ============================================================================
Eigen::Matrix3d Compute_FF(int PD, double d, const std::string& DEFflag) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(); // Identity matrix
    Eigen::Matrix3d FF = I; // Default: no deformation

    if (DEFflag == "EXT") {
        // Extension: stretch in x-direction only
        FF(0,0) = 1 + d;
    }
    else if (DEFflag == "EXP") {
        // Expansion: uniform scaling in all directions
        FF = (1 + d) * I;
    }
    else if (DEFflag == "SHR") {
        // Shear: shear deformation in y-direction
        FF(1,1) = d;
    }
    // If DEFflag doesn't match, FF remains identity (no deformation)
    
    return FF;
}

// ============================================================================
// FREE ALL POINTS: Set all points to free DOFs with zero force (homogeneous Neumann)
// ============================================================================
std::vector<Point> FreeAllPoints(std::vector<Point> PL) {
    int NoPs = PL.size();
    int PD = PL[0].PD;
    
    // Create default BC arrays: all DOFs free (1) with zero prescribed values
    std::vector<int> BCflg(PD, 1);
    std::vector<double> BCval(PD, 0.0);

    for (int i = 0; i < NoPs; i++) {
        PL[i].BCflg = BCflg;
        PL[i].BCval = BCval;
    }
    
    return PL;
}

// ============================================================================
// ASSIGN GLOBAL DOF: Assign global degree of freedom numbers to free DOFs
// ============================================================================
std::pair<std::vector<Point>, int> AssignGlobalDOF(std::vector<Point> PL) {
    int NoPs = PL.size();
    int PD = PL[0].PD;
    int total_DOFs = 0;
    
    std::cout << "\n=== AssignGlobalDOF DEBUG (Corrected) ===" << std::endl;
    
    for (int i = 0; i < NoPs; i++) {
        std::vector<int> DOF(PD, 0); // Initialize with 0
        for (int p = 0; p < PD; p++) {
            // ONLY assign a global index if the point is free to move
            if (PL[i].BCflg[p] == 1) {
                total_DOFs++;
                DOF[p] = total_DOFs;
            } else {
                // Fixed directions get index 0 (or -1), meaning "no DOF"
                DOF[p] = 0; 
            }
        }
        PL[i].DOF = DOF;
    }
    
    std::cout << "Free DOFs assigned: " << total_DOFs << std::endl;
    std::cout << "Total Points: " << NoPs << std::endl;
    
    return {PL, total_DOFs};
}

// ============================================================================
// ASSIGN BCS: Assign boundary conditions based on domain corners and flags
// ============================================================================
std::pair<std::vector<Point>, int> AssignBCs(const std::vector<Eigen::Vector3d>& Corners, 
                                            std::vector<Point> PL, const Eigen::Matrix3d& FF, 
                                            const std::string& BCflag, const std::string& PatchFlag) {
    int NoPs = PL.size();
    int PD = PL[0].PD;

    if (PD == 2) {
        // Extract 2D corner points
        Eigen::Vector3d A = Corners[0]; // Bottom-left
        Eigen::Vector3d B = Corners[1]; // Bottom-right
        Eigen::Vector3d C = Corners[2]; // Top-right
        Eigen::Vector3d D = Corners[3]; // Top-left

        if (BCflag == "STD") {
            // Standard boundary conditions
            PL = FreeAllPoints(PL); // Start with all points free
            double tol = 1e-3;

            if (PatchFlag == "fullpatch") {
                // Apply BCs to all boundary points
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    // Check if point is on domain boundary
                    if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol || 
                        (X(1)-A(1)) < tol || (X(1)-D(1)) > -tol) {
                        std::vector<int> BCflg(PD, 0); // Constrain all DOFs
                        Eigen::Vector3d BCval_vec = FF * X - X; // Prescribed displacement
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
            else if (PatchFlag == "horzpatch") {
                // Apply BCs only to horizontal boundaries
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol) {
                        std::vector<int> BCflg(PD, 0);
                        Eigen::Vector3d BCval_vec = FF * X - X;
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
            else if (PatchFlag == "vertpatch") {
                // Apply BCs only to vertical boundaries
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    if ((X(1)-A(1)) < tol || (X(1)-D(1)) > -tol) {
                        std::vector<int> BCflg(PD, 0);
                        Eigen::Vector3d BCval_vec = FF * X - X;
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
        }
        else if (BCflag == "DBC") {
            // Displacement boundary conditions on full boundary
            PL = FreeAllPoints(PL);
            double tol = 1e-3;

            for (int i = 0; i < NoPs; i++) {
                Eigen::Vector3d X = PL[i].X;
                if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol || 
                    (X(1)-A(1)) < tol || (X(1)-D(1)) > -tol) {
                    std::vector<int> BCflg(PD, 0);
                    Eigen::Vector3d BCval_vec = FF * X - X;
                    std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                    
                    PL[i].BCflg = BCflg;
                    PL[i].BCval = BCval;
                }
            }
        }
    }
    else if (PD == 3) {
        // 3D boundary conditions
        Eigen::Vector3d A = Corners[0]; // Left-south-bottom
        Eigen::Vector3d B = Corners[1]; // Right-south-bottom
        Eigen::Vector3d C = Corners[2]; // Right-north-bottom
        Eigen::Vector3d D = Corners[3]; // Left-north-bottom
        Eigen::Vector3d E = Corners[4]; // Left-south-top
        Eigen::Vector3d F = Corners[5]; // Right-south-top
        Eigen::Vector3d G = Corners[6]; // Right-north-top
        Eigen::Vector3d H = Corners[7]; // Left-north-top

        if (BCflag == "STD") {
            PL = FreeAllPoints(PL);
            double tol = 1e-3;

            if (PatchFlag == "fullpatch") {
                // Apply BCs to all boundary points in 3D
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    // Check if point is on any of the 6 domain boundaries
                    if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol || 
                        (X(1)-A(1)) < tol || (X(1)-D(1)) > -tol || 
                        (X(2)-A(2)) < tol || (X(2)-E(2)) > -tol) {
                        std::vector<int> BCflg(PD, 0);
                        Eigen::Vector3d BCval_vec = FF * X - X;
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
            else if (PatchFlag == "horzpatch") {
                // Apply BCs only to x-direction boundaries
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol) {
                        std::vector<int> BCflg(PD, 0);
                        Eigen::Vector3d BCval_vec = FF * X - X;
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
            else if (PatchFlag == "vertpatch") {
                // Apply BCs only to y-direction boundaries
                for (int i = 0; i < NoPs; i++) {
                    Eigen::Vector3d X = PL[i].X;
                    if ((X(1)-A(1)) < tol || (X(1)-D(1)) > -tol) {
                        std::vector<int> BCflg(PD, 0);
                        Eigen::Vector3d BCval_vec = FF * X - X;
                        std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                        
                        PL[i].BCflg = BCflg;
                        PL[i].BCval = BCval;
                    }
                }
            }
        }
        else if (BCflag == "DBC") {
            // Displacement boundary conditions on full 3D boundary
            PL = FreeAllPoints(PL);
            double tol = 1e-3;

            for (int i = 0; i < NoPs; i++) {
                Eigen::Vector3d X = PL[i].X;
                if ((X(0)-A(0)) < tol || (X(0)-B(0)) > -tol || 
                    (X(1)-A(1)) < tol || (X(1)-D(1)) > -tol || 
                    (X(2)-A(2)) < tol || (X(2)-E(2)) > -tol) {
                    std::vector<int> BCflg(PD, 0);
                    Eigen::Vector3d BCval_vec = FF * X - X;
                    std::vector<double> BCval(BCval_vec.data(), BCval_vec.data() + PD);
                    
                    PL[i].BCflg = BCflg;
                    PL[i].BCval = BCval;
                }
            }
        }
    }

    // Final assignment of global DOF numbers
    auto result = AssignGlobalDOF(PL);
    return result;
}

// ============================================================================
// Helper functions for calculating Residual and Stiffness (2D and 3D)
// ============================================================================

// 2D cross product helper (returns scalar)
double cross_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a(0) * b(1) - a(1) * b(0);
}

// 3D cross product helper (returns vector)
Eigen::Vector3d cross_3d(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Vector3d result;
    result(0) = a(1) * b(2) - a(2) * b(1);
    result(1) = a(2) * b(0) - a(0) * b(2);
    result(2) = a(0) * b(1) - a(1) * b(0);
    return result;
}

// ============================================================================
// ONE-NEIGHBOR ENERGY (psifunc1) - 2D and 3D
// ============================================================================
double psifunc1(const Eigen::VectorXd& XiI, const Eigen::VectorXd& xiI, double C1) {
    double l = xiI.norm();
    double L = XiI.norm();
    double s = (l - L) / L;
    double out = 0.5 * C1 * L * s * s;
    return out;
}

// ============================================================================
// TWO-NEIGHBOR ENERGY (psifunc2) - 2D and 3D
// ============================================================================
double psifunc2(const Eigen::VectorXd& XiI, const Eigen::VectorXd& XiII, 
                const Eigen::VectorXd& xiI, const Eigen::VectorXd& xiII, 
                double C2, double Delta, int PD) {
    double out = 0.0;
    double tol = 1e-8;
    
    double A = 0.0, a = 0.0;
    
    if (PD == 2) {
        Eigen::Vector2d XiI_2d = XiI.head<2>();
        Eigen::Vector2d XiII_2d = XiII.head<2>();
        Eigen::Vector2d xiI_2d = xiI.head<2>();
        Eigen::Vector2d xiII_2d = xiII.head<2>();
        A = std::abs(cross_2d(XiI_2d, XiII_2d));
        a = std::abs(cross_2d(xiI_2d, xiII_2d));
    } else if (PD == 3) {
        Eigen::Vector3d XiI_3d = XiI.head<3>();
        Eigen::Vector3d XiII_3d = XiII.head<3>();
        Eigen::Vector3d xiI_3d = xiI.head<3>();
        Eigen::Vector3d xiII_3d = xiII.head<3>();
        A = cross_3d(XiI_3d, XiII_3d).norm();
        a = cross_3d(xiI_3d, xiII_3d).norm();
    }
    
    if (A > tol && (XiI - XiII).norm() < Delta) {
        double s = (a - A) / A;
        out = 0.5 * C2 * A * s * s;
    }
    
    return out;
}

// ============================================================================
// THREE-NEIGHBOR ENERGY (psifunc3) - 3D only
// ============================================================================
double psifunc3(const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& XiIII,
                const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const Eigen::Vector3d& xiIII,
                double C3, double Delta) {
    double out = 0.0;
    double tol = 1e-8;
    
    double V = XiI.dot(cross_3d(XiII, XiIII));
    double v = xiI.dot(cross_3d(xiII, xiIII));
    
    if (std::abs(V) > tol && (XiI - XiII).norm() < Delta && 
        (XiI - XiIII).norm() < Delta && (XiII - XiIII).norm() < Delta) {
        double s = (std::abs(v) - std::abs(V)) / std::abs(V);
        out = 0.5 * C3 * std::abs(V) * s * s;
    }
    
    return out;
}

// ============================================================================
// ONE-NEIGHBOR FORCE (PP1) - 2D and 3D
// ============================================================================
Eigen::VectorXd PP1(const Eigen::VectorXd& XiI, const Eigen::VectorXd& xiI, double C1) {
    double l = xiI.norm();
    double L = XiI.norm();
    double s = (l - L) / L;
    Eigen::VectorXd eta = xiI / l;
    Eigen::VectorXd out = C1 * eta * s;
    return out;
}

// ============================================================================
// TWO-NEIGHBOR FORCE (PP2) - 2D and 3D
// ============================================================================
Eigen::VectorXd PP2(const Eigen::VectorXd& XiI, const Eigen::VectorXd& XiII,
                    const Eigen::VectorXd& xiI, const Eigen::VectorXd& xiII,
                    double C2, double Delta, int PD) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(PD);
    // std::cout << "\n=== DEBUG PP2 ===" << std::endl;
    // std::cout << "XiI: " << XiI.transpose() << std::endl;
    // std::cout << "XiII: " << XiII.transpose() << std::endl;
    // std::cout << "xiI: " << xiI.transpose() << std::endl;
    // std::cout << "xiII: " << xiII.transpose() << std::endl;
    
    double norm_A = 0.0, norm_a = 0.0;
    double tol = 1e-8;
    
    double A = 0.0, a = 0.0;
    
    if (PD == 2) {
        Eigen::Vector2d XiI_2d = XiI.head<2>();
        Eigen::Vector2d XiII_2d = XiII.head<2>();
        Eigen::Vector2d xiI_2d = xiI.head<2>();
        Eigen::Vector2d xiII_2d = xiII.head<2>();
        double cross_X = cross_2d(XiI_2d, XiII_2d);
        double cross_x = cross_2d(xiI_2d, xiII_2d);
        
        // std::cout << "2D cross(XiI, XiII) = " << cross_X << std::endl;
        // std::cout << "2D cross(xiI, xiII) = " << cross_x << std::endl;
        
        A = std::abs(cross_X);
        a = std::abs(cross_x);

        // std::cout << "A = |cross| = " << A << std::endl;
        // std::cout << "a = |cross| = " << a << std::endl;
    } else if (PD == 3) {
        Eigen::Vector3d XiI_3d = XiI.head<3>();
        Eigen::Vector3d XiII_3d = XiII.head<3>();
        Eigen::Vector3d xiI_3d = xiI.head<3>();
        Eigen::Vector3d xiII_3d = xiII.head<3>();
        A = cross_3d(XiI_3d, XiII_3d).norm();
        a = cross_3d(xiI_3d, xiII_3d).norm();
    }

    // std::cout << "A > tol? " << (A > tol) << " (A=" << A << ", tol=" << tol << ")" << std::endl;
    // std::cout << "norm(XiI-XiII) = " << (XiI - XiII).norm() << std::endl;
    // std::cout << "norm(XiI-XiII) < Delta? " << ((XiI - XiII).norm() < Delta) 
    //           << " (Delta=" << Delta << ")" << std::endl;
    
    if (A > tol && (XiI - XiII).norm() < Delta) {
        // std::cout << "ENTERING PP2 CALCULATION!" << std::endl;
        // std::cout << "G = 1/A - 1/a = " << (1.0/A) << " - " << (1.0/a) << std::endl;
        double G = (1.0 / A) - (1.0 / a);
        Eigen::VectorXd H = (xiII.dot(xiII) * xiI) - (xiII.dot(xiI) * xiII);
        out = 2.0 * C2 * G * H;

    }
    else{
        out = Eigen::VectorXd::Zero(PD); // Return zero force if collinear
    }
    
    return out;
}

// ============================================================================
// THREE-NEIGHBOR FORCE (PP3) - 3D only
// ============================================================================
Eigen::Vector3d PP3(const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& XiIII,
                    const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const Eigen::Vector3d& xiIII,
                    double C3, double Delta) {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    double tol = 1e-8;
    
    double V = XiI.dot(cross_3d(XiII, XiIII));
    double v = xiI.dot(cross_3d(xiII, xiIII));
    
    if (std::abs(V) > tol && std::abs(v) > tol && 
        (XiI - XiII).norm() < Delta && 
        (XiI - XiIII).norm() < Delta && 
        (XiII - XiIII).norm() < Delta) {
        
        // FIXED: Use consistent formula
        double factor = 3.0 * C3 * (1.0 / std::abs(V) - 1.0 / std::abs(v)) * v;
        out = factor * cross_3d(xiII, xiIII);
        
        // Debug NaN check
        if (std::isnan(out.norm())) {
            std::cout << "PP3 NaN: V=" << V << ", v=" << v 
                      << ", factor=" << factor << ", C3=" << C3 << std::endl;
        }
    }
    
    return out;
}

// ============================================================================
// ONE-NEIGHBOR STIFFNESS (AA1) - 2D and 3D
// ============================================================================
Eigen::MatrixXd AA1(const Eigen::VectorXd& XiI, const Eigen::VectorXd& xiI, double C1, int PD) {
    double l = xiI.norm();
    double L = XiI.norm();
    
    Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
    double s = (l - L) / L;
    Eigen::VectorXd eta = xiI / l;
    Eigen::MatrixXd eta_dyad_eta = eta * eta.transpose();
    
    Eigen::MatrixXd out = C1 * ((s / l) * (II - eta_dyad_eta) + (1.0 / L) * eta_dyad_eta);
    return out;
}

// ============================================================================
// TWO-NEIGHBOR STIFFNESS (AA2) - 2D and 3D
// ============================================================================
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AA2(const Eigen::VectorXd& XiI, const Eigen::VectorXd& XiII,
                                                  const Eigen::VectorXd& xiI, const Eigen::VectorXd& xiII,
                                                  double C2, double Delta, int PD) {
    Eigen::MatrixXd outI = Eigen::MatrixXd::Zero(PD, PD);
    Eigen::MatrixXd outJ = Eigen::MatrixXd::Zero(PD, PD);
    double tol = 1e-12; // Tighter tolerance
    
    // 1. Calculate Cross Products
    // In 2D, these are scalars representing the Z-component of the vector cross product
    double A = cross_2d(XiI.head<2>(), XiII.head<2>());
    double a = cross_2d(xiI.head<2>(), xiII.head<2>());
    
    double abs_A = std::abs(A);
    double abs_a = std::abs(a);

    // 2. Geometry Check (Mirror MATLAB AA > tol)
    if (abs_A > tol && (XiI - XiII).norm() < Delta) {
        Eigen::MatrixXd II = Eigen::MatrixXd::Identity(PD, PD);
        
        // 3. Construct helper matrices (Directly matching MATLAB BBI1, BBJ1)
        Eigen::MatrixXd BBI1 = (xiII.dot(xiII) * II) - (xiII * xiII.transpose());
        Eigen::MatrixXd BBJ1 = (2.0 * xiI * xiII.transpose()) - (xiI.dot(xiII) * II) - (xiII * xiI.transpose());
        
        // 4. Construct e-vectors
        Eigen::VectorXd eInII = (xiII.dot(xiII) * xiI) - (xiI.dot(xiII) * xiII);
        Eigen::VectorXd eIInI = (xiI.dot(xiI) * xiII) - (xiII.dot(xiI) * xiI);
        
        // 5. Final Assembly (CHECK SIGNS)
        // MATLAB: outI = 2 * CC * (1/AA - 1/aa) * BBI1 + 2 * CC * 1/(aa^3) * (eInII * eInII')
        // NOTE: Use AA and aa (the norms) as in your MATLAB code
        double term1_factor = 2.0 * C2 * (1.0 / abs_A - 1.0 / abs_a);
        double term2_factor = 2.0 * C2 / std::pow(abs_a, 3.0);

        outI = term1_factor * BBI1 + term2_factor * (eInII * eInII.transpose());
        outJ = term1_factor * BBJ1 + term2_factor * (eInII * eIInI.transpose());
    }
    
    return {outI, outJ};
}

// ============================================================================
// THREE-NEIGHBOR STIFFNESS (AA3) - 3D only
// ============================================================================
std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d> AA3(
    const Eigen::Vector3d& XiI, const Eigen::Vector3d& XiII, const Eigen::Vector3d& XiIII,
    const Eigen::Vector3d& xiI, const Eigen::Vector3d& xiII, const Eigen::Vector3d& xiIII,
    double C3, double Delta) {
    
    Eigen::Matrix3d outI = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d outJ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d outK = Eigen::Matrix3d::Zero();
    double tol = 1e-8;
    
    double V = XiI.dot(cross_3d(XiII, XiIII));
    double v = xiI.dot(cross_3d(xiII, xiIII));
    
    if (std::abs(V) > tol && (XiI - XiII).norm() < Delta && 
        (XiI - XiIII).norm() < Delta && (XiII - XiIII).norm() < Delta) {
        
        Eigen::Matrix3d II = Eigen::Matrix3d::Identity();
        
        Eigen::Vector3d cross_II_III = cross_3d(xiII, xiIII);
        Eigen::Matrix3d BBI1 = cross_II_III * cross_II_III.transpose();
        
        // Skew-symmetric matrix for xiIII
        Eigen::Matrix3d BBJ1;
        BBJ1 << 0, xiIII(2), -xiIII(1),
               -xiIII(2), 0, xiIII(0),
                xiIII(1), -xiIII(0), 0;
        
        Eigen::Vector3d cross_III_I = cross_3d(xiIII, xiI);
        Eigen::Matrix3d BBJ2 = cross_II_III * cross_III_I.transpose();
        
        // Skew-symmetric matrix for xiII
        Eigen::Matrix3d BBK1;
        BBK1 << 0, xiII(2), -xiII(1),
               -xiII(2), 0, xiII(0),
                xiII(1), -xiII(0), 0;
        
        Eigen::Vector3d cross_I_II = cross_3d(xiI, xiII);
        Eigen::Matrix3d BBK2 = cross_II_III * cross_I_II.transpose();
        
        outI = 3.0 * C3 * (1.0 / std::abs(V)) * BBI1;
        outJ = 3.0 * C3 * (1.0 / std::abs(V) - 1.0 / std::abs(v)) * v * BBJ1 + 3.0 * C3 * (1.0 / std::abs(V)) * BBJ2;
        outK = -3.0 * C3 * (1.0 / std::abs(V) - 1.0 / std::abs(v)) * v * BBK1 + 3.0 * C3 * (1.0 / std::abs(V)) * BBK2;
    }
    
    return {outI, outJ, outK};
}
// ============================================================================
// CALCULATE_RK: Calculate residual and stiffness for all points
// Direct conversion from MATLAB compute_residual and compute_stiffness
// ============================================================================

void calculate_rk(std::vector<Point>& PL, double C1, double C2, double Delta, int PD)
{
    // 1. Global iteration tracker (increments every time Newton-Raphson calls this)
    static int global_iter_count = 0;
    const double tol = 1e-8;
    (void)tol; // if unused for now

    // Parallelize over points (safe: each point writes only to itself)
    #pragma omp parallel for if(_OPENMP)
    for (int p = 0; p < static_cast<int>(PL.size()); ++p)
    {
        auto& point = PL[p];

        // -------------------- Load point data --------------------
        const int Nr = point.Nr;
        const int NI = point.NI;
        const int NInII = point.NInII;
        const int NInIInIII = point.NInIInIII;

        const double AV = point.AV;
        const Eigen::Vector3d X = point.X;
        const Eigen::Vector3d x = point.x;

        const auto& neighbors   = point.neighbors;
        const auto& neighborsX  = point.neighborsX_vec;
        const auto& neighborsx  = point.neighborsx_vec;

        // Material parameter C3
        const double C3 = (point.MatPars.size() > 2) ? point.MatPars[2] : 0.0;

        const int NNgbr = static_cast<int>(neighbors.size());

        // -------------------- Weighting factors --------------------
        double JI = 0.0, JInII = 0.0, JInIInIII = 0.0;

        if (PD == 2) {
            const double A = AV;
            JI    = (NI    > 0) ? (A / NI) : 0.0;
            JInII = (NInII > 0) ? ((A * A) / NInII) : 0.0;
        } else if (PD == 3) {
            const double V = AV;
            JI        = (NI          > 0) ? (V / NI) : 0.0;
            JInII     = (NInII       > 0) ? ((V * V) / NInII) : 0.0;
            JInIInIII = (NInIInIII   > 0) ? ((V * V * V) / NInIInIII) : 0.0;
        } else {
            // Unsupported PD
            continue;
        }

        // -------------------- Precompute relative vectors Xi, xi --------------------
        // Huge: avoid recomputing neighborsX[i]-X and neighborsx[i]-x in every loop.
        std::vector<Eigen::Vector3d> Xi3(NNgbr), xi3(NNgbr);
        for (int i = 0; i < NNgbr; ++i) {
            Xi3[i] = neighborsX[i] - X;
            xi3[i] = neighborsx[i] - x;
        }

        // Reusable buffers to avoid repeated dynamic allocations in tight loops
        Eigen::VectorXd XiI(PD), xiI(PD), XiII(PD), xiII(PD);

        // ==================== ENERGY CALCULATION ====================
        double psi1 = 0.0, psi2 = 0.0, psi3 = 0.0;

        // 1-neighbor energy
        if (C1 != 0.0) {
            for (int i = 0; i < NNgbr; ++i) {
                for (int d = 0; d < PD; ++d) {
                    XiI(d) = Xi3[i](d);
                    xiI(d) = xi3[i](d);
                }
                psi1 += JI * psifunc1(XiI, xiI, C1);
            }
        }

        // 2-neighbor energy
        if (C2 != 0.0) {
            for (int i = 0; i < NNgbr; ++i) {
                for (int j = 0; j < NNgbr; ++j) {
                    if (j == i) continue;

                    for (int d = 0; d < PD; ++d) {
                        XiI(d)  = Xi3[i](d);
                        xiI(d)  = xi3[i](d);
                        XiII(d) = Xi3[j](d);
                        xiII(d) = xi3[j](d);
                    }
                    psi2 += JInII * psifunc2(XiI, XiII, xiI, xiII, C2, Delta, PD);
                }
            }
        }

        // 3-neighbor energy (3D only)
        if (C3 != 0.0 && PD == 3) {
            for (int i = 0; i < NNgbr; ++i) {
                for (int j = 0; j < NNgbr; ++j) {
                    if (j == i) continue;
                    for (int k = 0; k < NNgbr; ++k) {
                        if (k == i || k == j) continue;

                        psi3 += JInIInIII * psifunc3(Xi3[i], Xi3[j], Xi3[k],
                                                     xi3[i], xi3[j], xi3[k],
                                                     C3, Delta);
                    }
                }
            }
        }

        point.psi = psi1 + psi2 + psi3;

        // ==================== RESIDUAL CALCULATION ====================
        Eigen::VectorXd R = Eigen::VectorXd::Zero(PD);

        // 1-neighbor residual
        if (C1 != 0.0) {
            for (int i = 0; i < NNgbr; ++i) {
                for (int d = 0; d < PD; ++d) {
                    XiI(d) = Xi3[i](d);
                    xiI(d) = xi3[i](d);
                }
                R += JI * PP1(XiI, xiI, C1);

            }
        }

        // 2-neighbor residual
        if (C2 != 0.0) {
            for (int i = 0; i < NNgbr; ++i) {
                for (int j = 0; j < NNgbr; ++j) {
                    if (j == i) continue;

                    for (int d = 0; d < PD; ++d) {
                        XiI(d)  = Xi3[i](d);
                        xiI(d)  = xi3[i](d);
                        XiII(d) = Xi3[j](d);
                        xiII(d) = xi3[j](d);
                    }
                    R += JInII * PP2(XiI, XiII, xiI, xiII, C2, Delta, PD);

                }
            }
        }

        // 3-neighbor residual (3D only)
        if (C3 != 0.0 && PD == 3) {
            Eigen::Vector3d R3 = Eigen::Vector3d::Zero();
            for (int i = 0; i < NNgbr; ++i) {
                for (int j = 0; j < NNgbr; ++j) {
                    if (j == i) continue;
                    for (int k = 0; k < NNgbr; ++k) {
                        if (k == i || k == j) continue;

                        R3 += JInIInIII * PP3(Xi3[i], Xi3[j], Xi3[k],
                                              xi3[i], xi3[j], xi3[k],
                                              C3, Delta);
                    }
                }
            }
            R = R + R3;
        }

        point.residual.resize(PD);
        point.residual = R;

        // ==================== STIFFNESS CALCULATION ====================
// Create extended neighbor list (neighbors + self)
int NNgbrE = NNgbr + 1;
std::vector<int> neighborsE = neighbors;
neighborsE.push_back(p);  // Add self (Nr)

// Initialize stiffness matrix (PD*PD x NNgbrE)
Eigen::MatrixXd K = Eigen::MatrixXd::Zero(PD * PD, NNgbrE);

// 1-neighbor stiffness
if (C1 != 0.0) {
    for (int i = 0; i < NNgbr; i++) {
        Eigen::VectorXd XiI(PD);
        Eigen::VectorXd xiI(PD);
        for (int d = 0; d < PD; d++) {
            XiI(d) = neighborsX[i](d) - X(d);
            xiI(d) = neighborsx[i](d) - x(d);
        }
        
        Eigen::MatrixXd AA1I = AA1(XiI, xiI, C1, PD);
        for (int b = 0; b < NNgbrE; b++) {
            
            double factor_i = (neighbors[i] == neighborsE[b] ? 1.0 : 0.0) - 
                              (p == neighborsE[b] ? 1.0 : 0.0);
            
            Eigen::MatrixXd K1tmp = JI * AA1I * factor_i;
            
            // Flatten to column vector (COLUMN-MAJOR order - matching MATLAB (:))
            for (int col = 0; col < PD; col++) {
                for (int row = 0; row < PD; row++) {
                    K(row + col * PD, b) += K1tmp(row, col);
                }
            }
        }

    }
}

// 2-neighbor stiffness  
if (C2 != 0.0) {
    for (int i = 0; i < NNgbr; i++) {
        for (int j = 0; j < NNgbr; j++) {
            if (j != i) {  
                Eigen::VectorXd XiI(PD), XiII(PD);
                Eigen::VectorXd xiI(PD), xiII(PD);
                for (int d = 0; d < PD; d++) {
                    XiI(d) = neighborsX[i](d) - X(d);
                    xiI(d) = neighborsx[i](d) - x(d);
                    XiII(d) = neighborsX[j](d) - X(d);
                    xiII(d) = neighborsx[j](d) - x(d);
                }
                
                auto [AA2I, AA2J] = AA2(XiI, XiII, xiI, xiII, C2, Delta, PD);
                
                for (int b = 0; b < NNgbrE; b++) {
                    
                    double factor_i = (neighbors[i] == neighborsE[b] ? 1.0 : 0.0) - 
                                      (Nr == neighborsE[b] ? 1.0 : 0.0);
                    double factor_j = (neighbors[j] == neighborsE[b] ? 1.0 : 0.0) - 
                                      (Nr == neighborsE[b] ? 1.0 : 0.0);
                    
                    Eigen::MatrixXd K2tmp = JInII * (AA2I * factor_i + AA2J * factor_j);
                    
                    // Flatten to column vector
                    for (int col = 0; col < PD; col++) {
                        for (int row = 0; row < PD; row++) {
                            K(row + col * PD, b) += K2tmp(row, col);
                        }

                    }
                    
                }
            }
        }
    }
}

// Store stiffness matrix (flattened, column-major)
point.stiffness.resize(PD * PD * NNgbrE);
int idx = 0;
for (int col = 0; col < NNgbrE; col++) {
    for (int row = 0; row < PD * PD; row++) {
        point.stiffness[idx++] = K(row, col);
    }
}
// if (Nr == 23) {  // Your debug shows Point 22 has Nr=23
//     static bool first_time = true;
//     if (first_time) {
//         std::cout << "\n>>>> [DEBUG STIFFNESS POINT 23 - C++ ITERATION " << global_iter_count << "]" << std::endl;
//                 std::cout << "JI: " << JI << " | JInII: " << JInII << std::endl;
        
//         int self_col = NNgbr; 
//                 std::cout << "K-Self (PD*PD vector):" << std::endl;
//                 for (int row = 0; row < PD * PD; ++row) {
//                     std::cout << "  " << K(row, self_col) << std::endl;
//                 }
//                 std::cout << "<<<< END STIFFNESS DEBUG" << std::endl;
//             }
// }
// // ========== END DEBUG ==========
    }
}



// ============================================================================
// ASSEMBLY: Assemble global residual vector and stiffness matrix
// ============================================================================
void assembly(const std::vector<Point>& point_list, int DOFs, Eigen::VectorXd& R, 
              Eigen::SparseMatrix<double>& K, const std::string& flag) {
    if (flag == "residual") {
        R.setZero(); // Reset residual vector
        
        
        for (const auto& point : point_list) {
            int PD = point.PD;
            Eigen::VectorXd R_P = point.residual.head(PD) + point.F_ext.head(PD);
            const std::vector<int>& BCflg = point.BCflg;
            const std::vector<int>& DOF = point.DOF;
            
            for (int d = 0; d < PD; d++) {
                if (BCflg[d] == 1 && DOF[d] > 0) {
                    R(DOF[d] - 1) += R_P(d);
                }
            }
        }
        
    }
    else if (flag == "stiffness") {
    K.resize(DOFs, DOFs);
    K.setZero();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(point_list.size() * 16);

    // Use INDEXED loop to know array index
    for (size_t i = 0; i < point_list.size(); i++) {
        const auto& point = point_list[i];
        const std::vector<int>& BCflg_p = point.BCflg;
        const std::vector<int>& DOF_p = point.DOF;
        int PD = point.PD;
        
        // Recreate neighborsE EXACTLY as in calculate_rk
        std::vector<int> neighborsE = point.neighbors;  // Contains array indices
        neighborsE.push_back(i);  // Add array index i, NOT point.Nr!
        int NNgbrE = neighborsE.size();

        // For each free DOF
        for (int d_p = 0; d_p < PD; d_p++) {
            if (BCflg_p[d_p] == 1 && DOF_p[d_p] > 0) {
                int row_global = DOF_p[d_p] - 1;
                
                for (int q = 0; q < NNgbrE; q++) {
                    int nbr_idx = neighborsE[q];  // This is array index
                    
                    // Check bounds
                    if (nbr_idx < 0 || nbr_idx >= (int)point_list.size()) {
                        std::cerr << "ERROR: Invalid neighbor index " << nbr_idx 
                                  << " for point " << i << std::endl;
                        continue;
                    }
                    
                    const auto& nbr_point = point_list[nbr_idx];
                    const std::vector<int>& BCflg_q = nbr_point.BCflg;
                    const std::vector<int>& DOF_q = nbr_point.DOF;
                    
                    for (int d_q = 0; d_q < PD; d_q++) {
                        if (BCflg_q[d_q] == 1 && DOF_q[d_q] > 0) {
                            int col_global = DOF_q[d_q] - 1;
                            
                            // Get stiffness value - CORRECT INDEXING
                            // Stiffness stored as: PD*PD rows x NNgbrE columns
                            // Column-major within each block
                            int local_index = d_p + d_q * PD;  // Column-major in 2x2 block
                            int flat_index = local_index + q * (PD * PD);
                            
                            if (flat_index >= 0 && flat_index < (int)point.stiffness.size()) {
                                double Kval = point.stiffness[flat_index];
                                triplets.emplace_back(row_global, col_global, Kval);
                            }
                        }
                    }
                }
            }
        }
    }
    
    K.setFromTriplets(triplets.begin(), triplets.end());
}
}

// ============================================================================
// UPDATE POINTS: Update point positions and forces based on solution increment
// ============================================================================
void update_points(std::vector<Point>& PL, double LF, Eigen::VectorXd& dx, 
                  const std::string& Update_flag, double F_prescribed, int number_of_right_patches) {
    int NoPs = PL.size();
    
    if (Update_flag == "Displacement") {
        // Update constrained displacement nodes (BCflg = 0) based on load factor
        for (int i = 0; i < NoPs; i++) {
            std::vector<int> BCflg = PL[i].BCflg;
            std::vector<double> BCval = PL[i].BCval;
            
            for (int d = 0; d < BCflg.size(); d++) {
                if (BCflg[d] == 0) {
                    // x = X + LF * u_prescribed (where u_prescribed = BCval)
                    PL[i].x(d) = PL[i].X(d) + (LF * BCval[d]); 
                }
            }
        }
    }
    else if (Update_flag == "Force") {
        // Update external force F_ext for force-prescribed boundary nodes
        for (int i = 0; i < NoPs; i++) {
            if (PL[i].Flag == "Right Patch") {
                // Distribute total prescribed force F_prescribed among all right patch points
                PL[i].F_ext.setZero();
                PL[i].F_ext(0) = LF * F_prescribed / number_of_right_patches;
            }
        }
    }
    else if (Update_flag == "Calculated") {
        // Update free nodes (BCflg = 1) with the calculated displacement increment dx
        for (int i = 0; i < NoPs; i++) {
            std::vector<int> BCflg = PL[i].BCflg;
            std::vector<int> DOF = PL[i].DOF;
            
            for (int d = 0; d < BCflg.size(); d++) {
                if (BCflg[d] == 1 && DOF[d] > 0) {
                    PL[i].x(d) += dx(DOF[d] - 1); // Add displacement increment
                }
            }
        }
    }

    // Update neighbor coordinates (neighborsx) based on the new current positions
    for (int i = 0; i < NoPs; i++) {
        for (size_t n = 0; n < PL[i].NI; n++) {
            int nbr_idx = PL[i].neighbors[n];
            // Access the updated 'x' coordinate of the neighbor point
            PL[i].neighborsx_vec[n] = PL[nbr_idx].x;
        }
    }
}
