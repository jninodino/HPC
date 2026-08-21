cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
project(ParallelLatticeBoltzmannSolver LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies: MPI, Eigen3, Kokkos
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    add_compile_definitions(WITH_MPI)
    message(STATUS "MPI found")
endif()

find_package(Eigen3 5.0.1)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      eigen3
      URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    )
    set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

find_package(Kokkos 4.7.03 EXACT)
if(NOT Kokkos_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.03.tar.gz
    )
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)

    option(LBM_ENABLE_CUDA "Enable CUDA backend" OFF)

    if(LBM_ENABLE_CUDA)
        message(STATUS "Building with CUDA backend")

        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    else()
        message(STATUS "Building with CPU Threads backend")

        set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_THREADS ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(kokkos)
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(executables)
add_subdirectory(tests)