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
SET(Trilinos_PREFIX "/global/cfs/cdirs/m4293/Trillions/install")
SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})

#include_directories("/global/cfs/cdirs/m4293/Trillions/install/include")
#link_directories("/global/cfs/cdirs/m4293/Trillions/install/lib64")
FIND_PACKAGE(Trilinos REQUIRED)

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

# Make sure to use same compilers and flags as Trilinos
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

SET(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
SET(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")



message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

SET(BUILD_TESTS ON)

#if (BUILD_TESTS)
#    add_subdirectory(tests)
#    message(STATUS "Building tests ....")
#endif()

SET(SOURCES cpp/vector_test.cpp)

add_executable(trillions_test ${SOURCES})

target_compile_options(trillions_test PRIVATE -std=c++17 -Ofast -march=native -Icpp -Icpp/lib -DMKL_ILP64 -m64 -fopenmp -I$ENV{MKLROOT}/include -lgomp)

target_link_libraries(trillions_test PRIVATE  MPI::MPI_C OpenMP::OpenMP_CXX)
