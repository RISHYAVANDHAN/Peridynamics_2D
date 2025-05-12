cmake_minimum_required(VERSION 3.10)
project(Peridynamics_2D)

set(CMAKE_CXX_STANDARD 17)

# Include Eigen
add_subdirectory(external/eigen)

# Your executable
add_executable(peridynamics_2d
        main.cpp
        Points.cpp
)

# Link Eigen (Header-only — no linking needed, just include dirs)
target_include_directories(peridynamics_2d PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        external/eigen
)

# Optional: include your project-specific headers
target_include_directories(peridynamics_2d PRIVATE ${CMAKE_SOURCE_DIR})
