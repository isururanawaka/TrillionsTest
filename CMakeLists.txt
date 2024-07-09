cmake_minimum_required(VERSION 3.23)
project(TrillionsTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

message("Finding OpenMP")
find_package(OpenMP REQUIRED)

# Trillions
message("Finding Trillions")
#find_path(TRILLIONS_INCLUDE_DIR trillions/trillions.h PATHS $ENV{TRILLIONS_ROOT}/include)
#find_library(TRILLIONS_LIBRARY Trillions PATHS $ENV{TRILLIONS_ROOT}/lib64)
#
#if (TRILLIONS_INCLUDE_DIR AND TRILLIONS_LIBRARY)
#    message(STATUS "Trillions include dir: ${TRILLIONS_INCLUDE_DIR}")
#    message(STATUS "Trillions library: ${TRILLIONS_LIBRARY}")
#else()
#    message(FATAL_ERROR "Could not find Trillions library or include directory.")
#endif()

include_directories("/global/cfs/cdirs/m4293/Trillions/install/include")
link_directories("/global/cfs/cdirs/m4293/Trillions/install/lib64")

message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

SET(BUILD_TESTS ON)

#if (BUILD_TESTS)
#    add_subdirectory(tests)
#    message(STATUS "Building tests ....")
#endif()

SET(SOURCES cpp/vector_test.hpp)

add_executable(trillions_test ${SOURCES})

target_compile_options(trillions_test PRIVATE -std=c++17 -Ofast -march=native -Icpp -Icpp/lib -DMKL_ILP64 -m64 -fopenmp -I$ENV{MKLROOT}/include -lgomp)

target_link_libraries(trillions_test PRIVATE Trillions MPI::MPI_C OpenMP::OpenMP_CXX)
