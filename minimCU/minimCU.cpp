// cl minimCU.cpp /EHsc /I%CUDA_PATH%\include cusolver.lib cusparse.lib cudart.lib /link/LIBPATH:%CUDA_PATH%\lib\x64

#pragma warning(disable:4834) // disable: "discarding return value of function with 'nodiscard' attribute"

#include <vector>
#include <Eigen/Core>
#include "cuSafeCall.h"

// Solves the least-square problem  x = argmin|A*z-b|  using cuSolver library
bool minimCU(int m, int n,
    double* A, /* matrix of size mxn */
    double* b, /* right-hand-side vector of size m */
    double* x, /* least-square solution vector of size n */
    double* min_norm /* minimum value of least-square problem */
)
{
    // Create device matrix and copy host to it
    double* d_A;
    gpuErrchk(cudaMalloc((void**)&d_A, m*n*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, A, m*n*sizeof(double), cudaMemcpyHostToDevice));

    // Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    int nnz = 0; // number of nonzero elements in sparse matrix

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseSafeCall(cusparseCreate(&handle));

    // Device side number of nonzero elements per row
    int* d_nnzPerVector;
    gpuErrchk(cudaMalloc((void**)&d_nnzPerVector, m*sizeof(int)));
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descrA, d_A, m, d_nnzPerVector, &nnz));
    // Host side number of nonzero elements per row
    int* h_nnzPerVector = (int*)malloc(m*sizeof(int));
    gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, m*sizeof(int), cudaMemcpyDeviceToHost));

    // Device side sparse matrix
    double* d_Asparse;
    gpuErrchk(cudaMalloc((void**)&d_Asparse, nnz*sizeof(double)));
    int* d_A_RowIndices;
    gpuErrchk(cudaMalloc((void**)&d_A_RowIndices, (m+1)*sizeof(int)));
    int* d_A_ColIndices;
    gpuErrchk(cudaMalloc((void**)&d_A_ColIndices, nnz*sizeof(int)));

    cusparseSafeCall(cusparseDdense2csr(handle, m, n, descrA, d_A, m, d_nnzPerVector, d_Asparse, d_A_RowIndices, d_A_ColIndices));

    // Host side sparse matrix
    double* h_Asparse = (double*)malloc(nnz*sizeof(double));
    int* h_A_RowIndices = (int*)malloc((m+1)*sizeof(int));
    int* h_A_ColIndices = (int*)malloc(nnz*sizeof(int));
    gpuErrchk(cudaMemcpy(h_Asparse, d_Asparse, nnz*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (m+1)*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz*sizeof(int), cudaMemcpyDeviceToHost));

    // CUDA solver initialization
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);

    int rankA;
    int* p = (int*)malloc(m*sizeof(int));

    cusolverStatus_t status = cusolverSpDcsrlsqvqrHost(solver_handle, m, n, nnz, descrA, h_Asparse, h_A_RowIndices, h_A_ColIndices, b, 0.000001, &rankA, x, p, min_norm);

    free(h_nnzPerVector);
    free(h_Asparse);
    free(h_A_RowIndices);
    free(h_A_ColIndices);
    free(p);

    cudaFree(d_A);
    cudaFree(d_nnzPerVector);
    cudaFree(d_Asparse);
    cudaFree(d_A_RowIndices);
    cudaFree(d_A_ColIndices);

    return (status == CUSOLVER_STATUS_SUCCESS);
}

#include <iostream>

int main()
{
    std::vector<Eigen::Vector3d> unknowns(50, Eigen::Vector3d(1, 1, 1));
    std::vector<Eigen::Vector3d> targets;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3d> deltas;
    for (int i = 0; i < unknowns.size(); ++i) {
        Eigen::Vector3d tgt(rand()%10 + 1, rand()%10 + 1, rand()%10 + 1);
        targets.push_back(tgt);
    }
    for (int i = 0; i < unknowns.size(); ++i) {
        Eigen::Vector3d norm(rand()%10 + 1, rand()%10 + 1, rand()%10 + 1);
        norm.normalize();
        normals.push_back(norm);
    }

    Eigen::MatrixXd fx(unknowns.size()*3, 1);
    for (int i = 0; i < unknowns.size(); ++i) {
        fx(i*3+0) = (targets[i](0) - unknowns[i](0)) * normals[i](0);
        fx(i*3+1) = (targets[i](1) - unknowns[i](1)) * normals[i](1);
        fx(i*3+2) = (targets[i](2) - unknowns[i](2)) * normals[i](2);
    }

    Eigen::MatrixXd J(unknowns.size()*3, 3);
    for (int i = 0; i < unknowns.size(); ++i) {
        J(i*3, 0) = -normals[i](0);  J(i*3+1, 0) = 0;               J(i*3+2, 0) = 0;
        J(i*3, 1) = 0;               J(i*3+1, 1) = -normals[i](1);  J(i*3+2, 1) = 0;
        J(i*3, 2) = 0;               J(i*3+1, 2) = 0;               J(i*3+2, 2) = -normals[i](2);
    }

    const int M = (int)unknowns.size()*3; // number of rows
    const int N = 3; // number of columns

    double* A = (double*)malloc(M*N*sizeof(double));
    for (int j = 0; j < N; j++)
       for (int i = 0; i < M; i++) {
       // column-major ordering
       A[i+j*M] = J(i,j);
    }

    double* b = (double*)malloc(M*sizeof(double));
    for (int i = 0; i < M; i++) b[i] = fx(i);

    double* x = (double*)malloc(N*sizeof(double));
    double min_norm;
    if ( minimCU(M, N, A, b, x, &min_norm) )
    {
        std::cout << "Solution:" << std::endl;
        for (int i = 0; i < N; i++) std::cout << x[i] << std::endl;;
        std::cout << "Minimum value: " << min_norm;
    }

    free(b); free(x); free(A);

    return 0;
}
