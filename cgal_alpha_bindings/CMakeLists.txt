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

# cgal_alpha_algo2 is the only module surface generation uses, and the only one
# in the default target. The rest are analysis aids, built by name on request.
pybind11_add_module(cgal_alpha_algo2 cgal_alpha_bindings_algo2.cpp)
pybind11_add_module(cgal_alpha_raw EXCLUDE_FROM_ALL cgal_alpha_raw.cpp)
pybind11_add_module(cgal_alpha_tagged EXCLUDE_FROM_ALL cgal_alpha_tagged.cpp)
pybind11_add_module(cgal_pmp_repair EXCLUDE_FROM_ALL cgal_pmp_repair.cpp)
pybind11_add_module(cgal_alpha_edge_analysis EXCLUDE_FROM_ALL cgal_alpha_edge_analysis.cpp)

target_link_libraries(cgal_alpha_algo2 PRIVATE CGAL::CGAL)
target_link_libraries(cgal_alpha_raw PRIVATE CGAL::CGAL)
target_link_libraries(cgal_alpha_tagged PRIVATE CGAL::CGAL)
target_link_libraries(cgal_pmp_repair PRIVATE CGAL::CGAL)
target_link_libraries(cgal_alpha_edge_analysis PRIVATE CGAL::CGAL)

# Optional targets, built only when SBL_ROOT points at an SBL checkout
# (https://sbl.inria.fr). Surface generation does not use them.
if(DEFINED ENV{SBL_ROOT})
    set(SBL_DIR "$ENV{SBL_ROOT}")

    pybind11_add_module(cgal_alpha EXCLUDE_FROM_ALL cgal_alpha_bindings.cpp)
    pybind11_add_module(cgal_patch_graph EXCLUDE_FROM_ALL cgal_patch_graph.cpp)
    pybind11_add_module(cgal_sbl_sampling EXCLUDE_FROM_ALL cgal_sbl_sampling.cpp)

    target_include_directories(cgal_sbl_sampling PRIVATE
        "${SBL_DIR}/Core/Union_of_balls_mesh_3/include"
        "${SBL_DIR}/Core/Betti_numbers/include"
        "${SBL_DIR}/Core/Union_of_balls_boundary_3/include"
        "${SBL_DIR}/Core/Spherical_kernel_extension_3/include"
        "${SBL_DIR}/Core/Triangulations_extensions_3/include"
        "${SBL_DIR}/Core/Union_find/include"
        "${SBL_DIR}/Core/Specialized_iterators/include"
    )
    target_link_libraries(cgal_sbl_sampling PRIVATE CGAL::CGAL)

    target_include_directories(cgal_alpha PRIVATE
        "${SBL_DIR}/Core/Union_of_balls_boundary_3/include"
        "${SBL_DIR}/Core/Triangulations_extensions_3/include"
        "${SBL_DIR}/Core/Union_find/include"
        "${SBL_DIR}/Core/Specialized_iterators/include"
    )
    target_include_directories(cgal_patch_graph PRIVATE
        "${SBL_DIR}/Core/Union_of_balls_boundary_3/include"
        "${SBL_DIR}/Core/Union_of_balls_surface_volume_3/include"
        "${SBL_DIR}/Core/Spherical_kernel_extension_3/include"
        "${SBL_DIR}/Core/Triangulations_extensions_3/include"
        "${SBL_DIR}/Core/Union_find/include"
        "${SBL_DIR}/Core/Specialized_iterators/include"
    )

    target_link_libraries(cgal_alpha PRIVATE CGAL::CGAL)

    find_library(SBL_MPFR_LIBRARY NAMES mpfr HINTS "$ENV{CONDA_PREFIX}/lib")
    find_library(SBL_GMP_LIBRARY NAMES gmp HINTS "$ENV{CONDA_PREFIX}/lib")
    if(NOT SBL_MPFR_LIBRARY OR NOT SBL_GMP_LIBRARY)
        message(FATAL_ERROR "cgal_patch_graph requires MPFR and GMP")
    endif()
    target_link_libraries(cgal_patch_graph PRIVATE
        CGAL::CGAL
        "${SBL_MPFR_LIBRARY}"
        "${SBL_GMP_LIBRARY}"
    )
endif()

# Print info
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "CGAL: ${CGAL_INCLUDE_DIRS}")
