cmake_minimum_required(VERSION 3.20)
project(Wannier_Coulomb)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add OpenMP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(SOURCES Wannier_Coulomb.cpp)
add_executable(Wannier_Coulomb.x ${SOURCES})

# Link OpenMP library
target_link_libraries(Wannier_Coulomb.x PUBLIC OpenMP::OpenMP_CXX)

