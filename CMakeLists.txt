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
find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra)

#include_directories("/global/cfs/cdirs/m4293/Trillions/install/include")
#link_directories("/global/cfs/cdirs/m4293/Trillions/install/lib64")

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
add_executable(TrillionsTest ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp)
target_link_libraries(TrillionsTest  Trilinos::all_selected_libs)

# Build the APP and link to Trilinos
add_executable(MyApp ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp)
target_link_libraries(MyApp  Trilinos::all_selected_libs)
# Or, above could have linked to just Tpetra::all_libs

# Set up a test
#enable_testing()
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
#        ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
#set(NUM_MPI_PROCS 4)
#add_test(MyAppTest mpicc -n ${NUM_MPI_PROCS} "${CMAKE_CURRENT_BINARY_DIR}/MyApp")
#set_tests_properties(MyAppTest PROPERTIES
#        PROCESSORS ${NUM_MPI_PROCS}
#        PASS_REGULAR_EXPRESSION "vec.norm1[(][)] = 40"
#        )
# NOTE: Above, mpiexec with mpich-3.2 requires you pass in the abs path to
# MyApp or mpiexec says it can't find it, even though it is running in the
# correct directory (see #10813).
