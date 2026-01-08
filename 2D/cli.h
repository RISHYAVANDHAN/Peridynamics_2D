#ifndef CLI_H
#define CLI_H

#include <string>

struct CLIOptions {
    double domain_size = 10.0;
    double Delta = 1.501;
    double L = 0.5;
    int number_of_patches = 3;
    int number_of_right_patches = 1;
    double C1 = 1.0;
    double C2 = 1.0;
    double nn = 2.0;
    double d = 0.1;
    double F_prescribed = 1.0;
    std::string Prescribed_Flag = "Displacement";  // or "Displacement"
    int steps = 100;
    double tol = 1e-10;
    std::string DEFflag = "EXT";
    std::string output_dir;
};

CLIOptions parseArguments(int argc, char* argv[]);

#endif // CLI_H
