#include <mpi.h>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Import.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
  // Initialize MPI
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  auto comm = Tpetra::getDefaultComm();

  // Check for correct number of arguments
  if (argc != 3) {
    if (comm->getRank() == 0) {
      std::cerr << "Usage: " << argv[0] << " <matrix_A.mtx> <matrix_B.mtx>" << std::endl;
    }
    return -1;
  }

  // Get file paths from arguments
  const char* matrixAPath = argv[1];
  const char* matrixBPath = argv[2];

  // Read sparse matrix market files
  using scalar_type = double;
  using local_ordinal_type = int;
  using global_ordinal_type = long long;
  using node_type = Tpetra::Map<>::node_type;

  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using crs_graph_type = Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type>;
  using map_type = Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type>;

  Teuchos::RCP<const map_type> mapA;
  Teuchos::RCP<crs_matrix_type> A;
  Teuchos::RCP<const map_type> mapB;
  Teuchos::RCP<crs_matrix_type> B;

  try {
    // Attempt to read matrix A as CrsMatrix
    A = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(matrixAPath, comm);
  } catch (const std::invalid_argument& e) {
    // If reading as CrsMatrix fails, try reading as CrsGraph
    try {
      Teuchos::RCP<crs_graph_type> graphA = Tpetra::MatrixMarket::Reader<crs_graph_type>::readSparseGraph(matrixAPath, comm);
      mapA = graphA->getRowMap();
      A = Teuchos::rcp(new crs_matrix_type(graphA));
    } catch (const std::exception& e_graph) {
      if (comm->getRank() == 0) {
        std::cerr << "Error reading matrix A from file: " << e_graph.what() << std::endl;
      }
      return -1;
    }
  }

  try {
    // Attempt to read matrix B as CrsMatrix
    B = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(matrixBPath, comm);
  } catch (const std::invalid_argument& e) {
    // If reading as CrsMatrix fails, try reading as CrsGraph
    try {
      Teuchos::RCP<crs_graph_type> graphB = Tpetra::MatrixMarket::Reader<crs_graph_type>::readSparseGraph(matrixBPath, comm);
      mapB = graphB->getRowMap();
      B = Teuchos::rcp(new crs_matrix_type(graphB));
    } catch (const std::exception& e_graph) {
      if (comm->getRank() == 0) {
        std::cerr << "Error reading matrix B from file: " << e_graph.what() << std::endl;
      }
      return -1;
    }
  }

  // Check dimensions
  if (A->getGlobalNumCols() != B->getGlobalNumRows()) {
    if (comm->getRank() == 0) {
      std::cerr << "Matrix dimensions do not match for multiplication." << std::endl;
    }
    return -1;
  }

  // Start timing
  Teuchos::Time timer("Matrix multiplication time");
  timer.start();

  // Perform matrix multiplication C = A * B
  Teuchos::RCP<crs_matrix_type> C = Teuchos::rcp(new crs_matrix_type(A->getRowMap(), 0));
  Tpetra::MatrixMatrix::Multiply(*A, false, *B, false, *C);

  // Stop timing
  timer.stop();
  double elapsedTime = timer.totalElapsedTime();

  // Print timing
  if (comm->getRank() == 0) {
    std::cout << "Matrix multiplication took " << elapsedTime << " seconds." << std::endl;
  }

  // Optionally write result to file
  if (comm->getRank() == 0) {
    Tpetra::MatrixMarket::Writer<crs_matrix_type>::writeSparseFile("matrix_C.mtx", C);
  }

  return 0;
}
