#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <iostream>

typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType node_type;
typedef double scalar_type;
typedef int local_ordinal_type;
typedef long long global_ordinal_type;

typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type> crs_matrix_type;
typedef Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type> crs_graph_type;
typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " matrixAPath matrixBPath" << std::endl;
    return 1;
  }

  const char *matrixAPath = argv[1];
  const char *matrixBPath = argv[2];

  Teuchos::RCP<crs_graph_type> graphA, graphB;
  try {
    graphA = Tpetra::MatrixMarket::Reader<crs_graph_type>::readSparseGraph(matrixAPath, comm);
    graphB = Tpetra::MatrixMarket::Reader<crs_graph_type>::readSparseGraph(matrixBPath, comm);
  } catch (std::exception &e) {
    std::cerr << "Error reading graph files: " << e.what() << std::endl;
    return 1;
  }

  // Create matrices A and B from the graphs, initializing values to 1
  Teuchos::RCP<crs_matrix_type> A = Teuchos::rcp(new crs_matrix_type(graphA));
  Teuchos::RCP<crs_matrix_type> B = Teuchos::rcp(new crs_matrix_type(graphB));
  A->setAllToScalar(1.0);
  B->setAllToScalar(1.0);
  A->fillComplete();
  B->fillComplete();

  // Multiply A and B, timing the operation
  Teuchos::RCP<crs_matrix_type> C;
  {
    Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("Matrix Multiplication Time"));
    C = Tpetra::MatrixMatrix::multiply(*A, false, *B, false);
  }

  // Print the timing results
  Teuchos::TimeMonitor::summarize();

  return 0;
}
