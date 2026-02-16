cmake_minimum_required(VERSION 3.16)
project(cgal_alpha_bindings)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Find CGAL (header-only, needs GMP/MPFR)
find_package(CGAL REQUIRED)

pybind11_add_module(cgal_alpha cgal_alpha_bindings.cpp)
target_link_libraries(cgal_alpha PRIVATE CGAL::CGAL)

# Print info
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "CGAL: ${CGAL_INCLUDE_DIRS}")
