cmake_minimum_required(VERSION 3.29)
project(Peridynamics_2D)

set(CMAKE_CXX_STANDARD 23)

# Find Eigen3 package
find_package(Eigen3 CONFIG REQUIRED)

# Define the executable target and source files
add_executable(Peridynamics_2D
        main.cpp
        Points.h
        Points.cpp)

# Link the Eigen3 library to your target
target_link_libraries(Peridynamics_2D PRIVATE Eigen3::Eigen)
