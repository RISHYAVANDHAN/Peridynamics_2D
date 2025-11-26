cmake_minimum_required(VERSION 3.14)
project(Peridynamics_2D)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Simply include Eigen headers
include_directories(eigen)

# Fix paths to source files
add_executable(Peridynamics_2D
    2D/hyperdual.h
    2D/main.cpp
    2D/Points.cpp
    2D/Points.h
    2D/cli.h
    2D/cli.cpp
    2D/logger.h
)