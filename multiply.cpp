#include <mpi.h>
#include <Epetra_MpiComm.h>
#include <Epetra_CrsMatrix.h>
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_RowMatrixOut.h>
#include <Epetra_Time.h>
#include <Teuchos_RCP.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // Check for correct number of arguments
  if (argc != 3) {
    if (Comm.MyPID() == 0) {
      std::cerr << "Usage: " << argv[0] << " <matrix_A.mtx> <matrix_B.mtx>" << std::endl;
    }
    MPI_Finalize();
    return -1;
  }

  // Get file paths from arguments
  const char* matrixAPath = argv[1];
  const char* matrixBPath = argv[2];

  // Read sparse matrix market files
  Teuchos::RCP<Epetra_CrsMatrix> A;
  Teuchos::RCP<Epetra_CrsMatrix> B;
  EpetraExt::MatrixMarketFileToCrsMatrix(matrixAPath, Comm, A);
  EpetraExt::MatrixMarketFileToCrsMatrix(matrixBPath, Comm, B);

  // Check dimensions
  if (A->NumGlobalCols() != B->NumGlobalRows()) {
    if (Comm.MyPID() == 0) {
      std::cerr << "Matrix dimensions do not match for multiplication." << std::endl;
    }
    MPI_Finalize();
    return -1;
  }

  // Start timing
  Epetra_Time timer(Comm);

  // Perform matrix multiplication C = A * B
  Teuchos::RCP<Epetra_CrsMatrix> C;
  EpetraExt::MatrixMatrix::Multiply(*A, false, *B, false, C);

  // Stop timing
  double elapsedTime = timer.ElapsedTime();

  // Print timing
  if (Comm.MyPID() == 0) {
    std::cout << "Matrix multiplication took " << elapsedTime << " seconds." << std::endl;
  }

  // Optionally write result to file
  if (Comm.MyPID() == 0) {
    EpetraExt::RowMatrixToMatrixMarketFile("matrix_C.mtx", *C);
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
