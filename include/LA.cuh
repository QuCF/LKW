#pragma once

#include "mix.h"


/**
 * ----------------------------------------------------------------------
 * --- Linear algebra ---
 * ----------------------------------------------------------------------
*/
class LA{
public:


    /**
     * Find SVD of the matrix @param A. 
     * The method does not work for large matrices.
    */
    static void cond_number_cusolver(YMatrix<ycomplex> &A)
    {
        float time;
        cudaEvent_t start, stop;
        int info_gpu = 0; /* host copy of error info */
        int *devInfo = nullptr;

        uint32_t N = A.get_nr();

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cusolverDnHandle_t cusolverH = NULL;
        cublasHandle_t cublasH = NULL;
        int32_t lda = N; 
        uint32_t N2 = N * N;

        std::shared_ptr<ycomplex[]> U  = std::shared_ptr<ycomplex[]>(new ycomplex[N2]); // left singular vectors;
        std::shared_ptr<ycomplex[]> VT = std::shared_ptr<ycomplex[]>(new ycomplex[N2]); // complex-conjugated right singular vectors;
        std::shared_ptr<double[]> S  = std::shared_ptr<double[]>(new double[N]); // numerical singular values ordered in 
                                                                                // the descending order;
        cuDoubleComplex *d_A  = nullptr;
        cuDoubleComplex *d_U  = nullptr;  
        cuDoubleComplex *d_VT = nullptr; 
        double *d_S = nullptr;  

        int lwork = 0; /* size of workspace */
        cuDoubleComplex *d_work = nullptr;
        double *d_rwork = nullptr;

        /* create cusolver handle, bind a stream */
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUBLAS_CHECK(cublasCreate(&cublasH));

        /* allocate matrices on the device */
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),  sizeof(cuDoubleComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U),  sizeof(cuDoubleComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(cuDoubleComplex) * N2));
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W),  sizeof(cuDoubleComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S),  sizeof(double) * N));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

        /* copy the matrix to the device */
        CUDA_CHECK(cudaMemcpy(d_A, A.get_1d_column_major(), sizeof(ycomplex) * N2, cudaMemcpyHostToDevice));

        /* query working space of SVD */
        CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(cusolverH, N, N, &lwork));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork));

        /* compute SVD*/
        signed char jobu  = 'N';  // all m columns of U
        signed char jobvt = 'N'; // all n columns of VT
        CUSOLVER_CHECK(cusolverDnZgesvd(
            cusolverH, jobu, jobvt, 
            N, N, d_A, lda, 
            d_S, 
            d_U,  lda, 
            d_VT, lda, 
            d_work, lwork, d_rwork, devInfo
        ));

        double s_min, s_max;
        CUDA_CHECK(cudaMemcpy(&s_max, &(d_S[0]),    sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&s_min, &(d_S[N-1]), sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "s-min = " << s_min << std::endl;
        std::cout << "s-max = " << s_max << std::endl;

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Done. Elapsed time:  %0.3e s \n", time/1e3);
        
        printf("after gesvd: info_gpu = %d\n", info_gpu);
        if (0 == info_gpu) {
            std::printf("gesvd converges \n");
        } else if (0 > info_gpu) {
            std::printf("--- WARNING: %d-th parameter is wrong --- \n", -info_gpu);
            exit(1);
        } else {
            std::printf("--- WARNING: info = %d : gesvd does NOT CONVERGE --- \n", info_gpu);
        }

        double cn = s_max / s_min;
        printf("\tResulting condition number: %0.3e\n", cn);

        /* free resources */
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_U));
        CUDA_CHECK(cudaFree(d_VT));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(devInfo));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_rwork));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
        CUBLAS_CHECK(cublasDestroy(cublasH));
        CUDA_CHECK(cudaDeviceReset());

        std::cout << std::endl;
    }


    /**
     * Solve Ax = b for a sparse A.
     * Input and output variables have to be in the device memory.
    */
   static int solve_sparse_system(
        const SpMatrixC& A, 
        cuDoubleComplex* b, 
        cuDoubleComplex*& x, 
        const double& tol = 1e-12
    ){
        cusolverSpHandle_t cusolverH = NULL;
        cusparseMatDescr_t descrA = NULL;
        int singularity = 0;
        int result = 0;

        CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

        /* Matrix description */
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
        CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 

        /* Solver */
        cusolverSpZcsrlsvqr(
                cusolverH,
                A.N,
                A.Nnz,
                descrA,
                A.values,
                A.rows,
                A.columns,
                b,
                tol,
                0,
                x,
                &singularity
        );

        if(singularity >= 0)
        {
            printf(
                "--> ERROR in LA::solve_sparse_system: singularity in R(%d,%d)\n", 
                singularity, 
                singularity
            );
            result = -1;
        }

        /* free resources */
        CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));
        CUDA_CHECK(cudaDeviceReset());
        return result;
   }




// /**
//  * Solve Ax = b for a sparse A.
//  * Input and output variables have to be in the device memory.
// */
// static int solve_sparse_system(
//     const SpMatrixC& A, 
//     cuDoubleComplex* b, 
//     cuDoubleComplex*& x, 
//     const double& tol = 1e-12
// ){
//     cusolverSpHandle_t cusolverH = NULL;
//     cusparseMatDescr_t descrA = NULL;
//     int singularity = 0;
//     int result = 0;

//     // GPU does batch QR
//     // d_A is CSR format, d_csrValA is of size nnzA*batchSize
//     // d_x is a matrix of size batchSize * m
//     // d_b is a matrix of size batchSize * m
//     // int *d_csrRowPtrA = nullptr;
//     // int *d_csrColIndA = nullptr;
//     // double *d_csrValA = nullptr;
//     // double *d_b = nullptr; // batchSize * m
//     // double *d_x = nullptr; // batchSize * m

//     /*      
//     *      | 1                |
//     *  A = |       2          |
//     *      |            3     |
//     *      | 0.1  0.1  0.1  4 |
//     *  CSR of A is based-1
//     *
//     *  b = [1 1 1 1]
//     */
//     // const int m = 4;
//     // const int nnzA = 7;
//     // const std::vector<int> csrRowPtrA = {1, 2, 3, 4, 8};
//     // const std::vector<int> csrColIndA = {1, 2, 3, 1, 2, 3, 4};
//     // const std::vector<double> csrValA = {1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
//     // const std::vector<double> b = {1.0, 1.0, 1.0, 1.0};

//     // std::vector<double> csrValABatch(nnzA * batchSize, 0);
//     // std::vector<double> bBatch(m * batchSize, 0);
//     // std::vector<double> xBatch(m * batchSize, 0);

//     // step 1: prepare Aj and bj on host
//     //  Aj is a small perturbation of A
//     //  bj is a small perturbation of b
//     //  csrValABatch = [A0, A1, A2, ...]
//     //  bBatch = [b0, b1, b2, ...]
//     // for (int colidx = 0; colidx < nnzA; colidx++) {
//     //     double Areg = csrValA[colidx];
//     //     for (int batchId = 0; batchId < batchSize; batchId++) {
//     //         double eps = (static_cast<double>((std::rand() % 100) + 1)) * 1.e-4;
//     //         csrValABatch[batchId * nnzA + colidx] = Areg + eps;
//     //     }
//     // }

//     // for (int j = 0; j < m; j++) {
//     //     double breg = b[j];
//     //     for (int batchId = 0; batchId < batchSize; batchId++) {
//     //         double eps = (static_cast<double>((std::rand() % 100) + 1)) * 1.e-4;
//     //         bBatch[batchId * m + j] = breg + eps;
//     //     }
//     // }

//     CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

//     /* Matrix description */
//     CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
//     CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//     CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 


//     /* Solver */
//     cusolverSpZcsrlsvqr(
//             cusolverH,
//             A.N,
//             A.Nnz,
//             descrA,
//             A.values,
//             A.rows,
//             A.columns,
//             b,
//             tol,
//             0,
//             x,
//             &singularity
//     );

//     if(singularity >= 0)
//     {
//         printf(
//             "--> ERROR in LA::solve_sparse_system: singularity in R(%d,%d)\n", 
//             singularity, 
//             singularity
//         );
//         result = -1;
//     }

//     /* free resources */
//     CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));
//     CUDA_CHECK(cudaDeviceReset());
//     return result;
// }



// /************************************/
// /* SAVE REAL ARRAY FROM GPU TO FILE */
// /************************************/
// template <class T>
// void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

//     T *h_in = (T *)malloc(M * sizeof(T));

//     gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

//     std::ofstream outfile;
//     outfile.open(filename);
//     for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
//     outfile.close();

// }


// /***********************************************/
// /* FUNCTION TO COMPUTE THE COEFFICIENTS VECTOR */
// /***********************************************/
// void computeCoefficientsVector(const double * __restrict h_A, const double * __restrict h_xref, 
//                                double * __restrict h_y, const int N) {

//     for (int k = 0; k < N; k++) h_y[k] = 0.f;

//     for (int m = 0; m < N; m++)
//         for (int n = 0; n < N; n++)
//             h_y[m] = h_y[m] + h_A[n * N + m] * h_xref[n];

// }

// /************************************/
// /* COEFFICIENT REARRANGING FUNCTION */
// /************************************/
// void rearrange(double *vec, int *pivotArray, int N){
//     for (int i = 0; i < N; i++) {
//         double temp = vec[i];
//         vec[i] = vec[pivotArray[i] - 1];
//         vec[pivotArray[i] - 1] = temp;
//     }   
// }

// /**
//  * Solve the system AX = B, where A, B, X are dense matrices.
// */
// int dense_solver(YMatrix<ycomplex> &A, YMatrix<ycomplex> &X, YMatrix<ycomplex> &B)
// {
//     uint32_t N = A.get_nr();
//     uint32_t N2 = N * N;

//     // --- CUBLAS initialization ---
//     cublasHandle_t cublas_handle;
//     CUBLAS_CHECK(cublasCreate(&cublas_handle));

//     // --- Allocate device space for the input matrices 
//     cuDoubleComplex *d_A; 
//     cuDoubleComplex *d_X; 
//     cuDoubleComplex *d_B; 
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * N2));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(cuDoubleComplex) * N2));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cuDoubleComplex) * N2));
//     CUDA_CHECK(cudaMemcpy(d_A, A.get_1d_column_major(), sizeof(ycomplex) * N2, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_X, X.get_1d_column_major(), sizeof(ycomplex) * N2, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_B, B.get_1d_column_major(), sizeof(ycomplex) * N2, cudaMemcpyHostToDevice));

//     /**********************************/
//     /* COMPUTING THE LU DECOMPOSITION */
//     /**********************************/
//     timerLU.StartCounter();

//     // --- Creating the array of pointers needed as input/output to the batched getrf
//     double **h_inout_pointers = (double **)malloc(Nmatrices * sizeof(double *));
//     for (int i = 0; i < Nmatrices; i++) 
//         h_inout_pointers[i] = d_A + i * N * N;

//     double **d_inout_pointers;
//     gpuErrchk(cudaMalloc(&d_inout_pointers, Nmatrices * sizeof(double *)));
//     gpuErrchk(cudaMemcpy(d_inout_pointers, h_inout_pointers, Nmatrices * sizeof(double *), cudaMemcpyHostToDevice));
//     free(h_inout_pointers);

//     int *d_pivotArray; gpuErrchk(cudaMalloc(&d_pivotArray, N * Nmatrices * sizeof(int)));
//     int *d_InfoArray;  gpuErrchk(cudaMalloc(&d_InfoArray,      Nmatrices * sizeof(int)));

//     int *h_InfoArray  = (int *)malloc(Nmatrices * sizeof(int));

//     cublasSafeCall(cublasDgetrfBatched(cublas_handle, N, d_inout_pointers, N, d_pivotArray, d_InfoArray, Nmatrices));
//     //cublasSafeCall(cublasDgetrfBatched(cublas_handle, N, d_inout_pointers, N, NULL, d_InfoArray, Nmatrices));

//     gpuErrchk(cudaMemcpy(h_InfoArray, d_InfoArray, Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

//     for (int i = 0; i < Nmatrices; i++)
//         if (h_InfoArray[i] != 0) {
//             fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
//             cudaDeviceReset();
//             exit(EXIT_FAILURE);
//         }

//     timingLU = timerLU.GetCounter();
//     printf("Timing LU decomposition %f [ms]\n", timingLU);

//     /*********************************/
//     /* CHECKING THE LU DECOMPOSITION */
//     /*********************************/
//     saveCPUrealtxt(h_A,          "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\A.txt", N * N);
//     saveCPUrealtxt(h_y,          "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\y.txt", N);
//     saveGPUrealtxt(d_A,          "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\Adecomposed.txt", N * N);
//     saveGPUrealtxt(d_pivotArray, "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\pivotArray.txt", N);

//     /******************************************************************************/
//     /* APPROACH NR.1: COMPUTE THE INVERSE OF A STARTING FROM ITS LU DECOMPOSITION */
//     /******************************************************************************/
//     timerApproach1.StartCounter();

//     // --- Allocate device space for the inverted matrices 
//     double *d_Ainv; gpuErrchk(cudaMalloc(&d_Ainv, N * N * Nmatrices * sizeof(double)));

//     // --- Creating the array of pointers needed as output to the batched getri
//     double **h_out_pointers = (double **)malloc(Nmatrices * sizeof(double *));
//     for (int i = 0; i < Nmatrices; i++) h_out_pointers[i] = (double *)((char*)d_Ainv + i * ((size_t)N * N) * sizeof(double));

//     double **d_out_pointers;
//     gpuErrchk(cudaMalloc(&d_out_pointers, Nmatrices*sizeof(double *)));
//     gpuErrchk(cudaMemcpy(d_out_pointers, h_out_pointers, Nmatrices*sizeof(double *), cudaMemcpyHostToDevice));
//     free(h_out_pointers);

//     cublasSafeCall(
//         cublasDgetriBatched(
//             cublas_handle, N, (const double **)d_inout_pointers, 
//             N, d_pivotArray, d_out_pointers, N, d_InfoArray, Nmatrices
//             )
//         );

//     gpuErrchk(cudaMemcpy(h_InfoArray, d_InfoArray, Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

//     for (int i = 0; i < Nmatrices; i++)
//         if (h_InfoArray[i] != 0) {
//         fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
//         cudaDeviceReset();
//         exit(EXIT_FAILURE);
//         }

//     double alpha1 = 1.f;
//     double beta1 = 0.f;

//     cublasSafeCall(cublasDgemv(cublas_handle, CUBLAS_OP_N, N, N, &alpha1, d_Ainv, N, d_y, 1, &beta1, d_x, 1));

//     timingApproach1 = timingLU + timerApproach1.GetCounter();
//     printf("Timing approach 1 %f [ms]\n", timingApproach1);

//     /**************************/
//     /* CHECKING APPROACH NR.1 */
//     /**************************/
//     saveGPUrealtxt(d_x, "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\xApproach1.txt", N);

//     /*************************************************************/
//     /* APPROACH NR.2: INVERT UPPER AND LOWER TRIANGULAR MATRICES */
//     /*************************************************************/
//     timerApproach2.StartCounter();

//     double *d_P; gpuErrchk(cudaMalloc(&d_P, N * N * sizeof(double)));

//     gpuErrchk(cudaMemcpy(h_y, d_y, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));
//     int *h_pivotArray = (int *)malloc(N * Nmatrices*sizeof(int));
//     gpuErrchk(cudaMemcpy(h_pivotArray, d_pivotArray, N * Nmatrices * sizeof(int), cudaMemcpyDeviceToHost));

//     rearrange(h_y, h_pivotArray, N);
//     gpuErrchk(cudaMemcpy(d_y, h_y, N * Nmatrices * sizeof(double), cudaMemcpyHostToDevice));

//     // --- Now P*A=L*U
//     //     Linear system A*x=y => P.'*L*U*x=y => L*U*x=P*y

//     // --- 1st phase - solve Ly = b 
//     const double alpha = 1.f;

//     // --- Function solves the triangular linear system with multiple right hand sides, function overrides b as a result 

//     // --- Lower triangular part
//     cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, d_A, N, d_y, N));

//     // --- Upper triangular part
//     cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_y, N));

//     timingApproach2 = timingLU + timerApproach2.GetCounter();
//     printf("Timing approach 2 %f [ms]\n", timingApproach2);

//     /**************************/
//     /* CHECKING APPROACH NR.2 */
//     /**************************/
//     saveGPUrealtxt(d_y, "D:\\Project\\solveSquareLinearSystemCUDA\\solveSquareLinearSystemCUDA\\xApproach2.txt", N);

//     return 0;
// }







};





