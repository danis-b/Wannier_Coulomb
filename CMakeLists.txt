cmake_minimum_required(VERSION 3.0)
project(Wannier_Coulomb)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(SOURCES Wannier_Coulomb.cpp)
add_executable(Wannier_Coulomb.x ${SOURCES})
