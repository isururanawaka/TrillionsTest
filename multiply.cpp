#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <iostream>

typedef double scalar_type;
typedef int local_ordinal_type;
typedef long long global_ordinal_type;
typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type> crs_matrix_type;
typedef Tpetra::Map<local_ordinal_type, global_ordinal_type> map_type;

Teuchos::RCP<crs_matrix_type> readMatrixWithDefaultValues(const std::string& filePath, const Teuchos::RCP<const Teuchos::Comm<int>>& comm) {
  Teuchos::RCP<crs_matrix_type> matrix;
  try {
    matrix = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(filePath, comm);
  } catch (std::exception &e) {
    std::cerr << "Error reading matrix file: " << e.what() << std::endl;
    return Teuchos::null;
  }

  // Ensure all entries have value 1
  size_t numRows = matrix->getLocalNumRows();
  for (size_t i = 0; i < numRows; ++i) {
    local_ordinal_type localRow = static_cast<local_ordinal_type>(i);
    typename crs_matrix_type::local_inds_host_view_type indices;
    typename crs_matrix_type::values_host_view_type values;
    matrix->getLocalRowView(localRow, indices, values);

    for (size_t j = 0; j < indices.size(); ++j) {
      values(j) = 1.0;
    }
  }
  matrix->fillComplete();
  return matrix;
}

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " matrixAPath matrixBPath" << std::endl;
    return 1;
  }

  std::string matrixAPath = argv[1];
  std::string matrixBPath = argv[2];

  Teuchos::RCP<crs_matrix_type> A = readMatrixWithDefaultValues(matrixAPath, comm);
  Teuchos::RCP<crs_matrix_type> B = readMatrixWithDefaultValues(matrixBPath, comm);

  if (A.is_null() || B.is_null()) {
    return 1;
  }

  // Multiply A and B, timing the operation
  Teuchos::RCP<crs_matrix_type> C;
  {
    Teuchos::TimeMonitor tm(*Teuchos::TimeMonitor::getNewTimer("Matrix Multiplication Time"));
    Tpetra::MatrixMatrix::Multiply(*A, false, *B, false, C);
  }

  // Print the timing results
  Teuchos::TimeMonitor::summarize();

  return 0;
}
