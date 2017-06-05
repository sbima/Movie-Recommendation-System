#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> 
#include <cusolverDn.h>
#include </home/sbimavar/cuda/ALS/final/lin_equation_solve.cu>

void create_diagonal_matrix(double *Dmatrix, double * matrix, int array_length)
{   
	for(int i=0;i<array_length;i++)
    {
        for(int j=0;j<array_length;j++)
        {
            if(i==j)
                Dmatrix[j*array_length+i]=matrix[i];
            else
                Dmatrix[j*array_length+i]=0;    
        }       
    }  
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) 
{
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

__global__ void Add(double * dX_out, double * dX_in, double * dY_in)
{
	int idx = threadIdx.x;
	dX_out[idx] = dX_in[idx] + dY_in[idx];
}

__global__ void trans(double * dX_out, double * dX_in, int total_rows, int total_columns)
{		
	int idx = threadIdx.x;
	int row_number = idx / total_columns;
//	printf("%d \t", row_number);
	int col_number = idx - (row_number * total_columns);
//	printf("%d \t", col_number);
	int index = idx;
	int new_r = col_number;
	int new_c = row_number;
	int new_index = (new_r * total_rows) + new_c;
//	printf("%d \t", new_index);
	dX_out[new_index] = dX_in[index];
//	printf("%d \t", dX_out[new_index]);
}

void calling_trans(int nr_rows_A, int nr_cols_A, double * h_Ab, double * h_Ab_out)
{
	const int ARRAY_BYTES_x = nr_rows_A * nr_cols_A * sizeof(double);
	
	double * dX_in;
	double * dX_out;

        // allocate GPU memory
	cudaMalloc((void**) &dX_in, ARRAY_BYTES_x);
    cudaMalloc((void**) &dX_out, ARRAY_BYTES_x);

        // transfer the array to the GPU
	cudaMemcpy(dX_in, h_Ab, ARRAY_BYTES_x, cudaMemcpyHostToDevice);        
	
	trans<<<1, nr_rows_A * nr_cols_A>>>(dX_out, dX_in, nr_rows_A, nr_cols_A);
	
	cudaMemcpy(h_Ab_out, dX_out, ARRAY_BYTES_x, cudaMemcpyDeviceToHost);
	
	cudaFree(dX_in);
    cudaFree(dX_out);
}

void mul(int nr_rows_A, int nr_cols_A, int nr_rows_B, int nr_cols_B, int nr_rows_finalproduct, int nr_cols_finalproduct, double * h_Ab, double * h_Bb, double * final)
{
	//transpose of first matrix starts here
	double *h_Ab_out;
	h_Ab_out = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
	
	calling_trans(nr_rows_A, nr_cols_A, h_Ab, h_Ab_out);
	
	//transpose of second matrix starts here
	double *h_Bb_out;
	h_Bb_out = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
	
	calling_trans(nr_rows_B, nr_cols_B, h_Bb, h_Bb_out);
	
	//multiplication to get the resultant matrix starts here
		int nr_rows_product = nr_rows_A;
		int nr_cols_product = nr_cols_B;
	
	double *product = (double *)malloc(nr_rows_product * nr_cols_product * sizeof(double));

	// Allocate 3 arrays on GPU
	double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(double));
	cudaMalloc(&d_C,nr_rows_product * nr_cols_product * sizeof(double));

	// If you already have useful values in A and B you can copy them in GPU:
	cudaMemcpy(d_A,h_Ab_out,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_Bb_out,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice);
	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_product, nr_cols_A, nr_cols_product);
	
	cudaMemcpy(product,d_C,nr_rows_product * nr_cols_product * sizeof(double),cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
	
	// Now, taking the transpose of the resultant matrix (resultant matrix's dimensions (no.of.rows switched with no.of.cols)) gives us the final correct answer 
	
	calling_trans(nr_rows_finalproduct, nr_cols_finalproduct, product, final);
	
	free(h_Ab_out);
	free(h_Bb_out);
	free(product);
	
}

void als_solve_X(double *X, int u, double *Y, double *Wu, int n_factors, double lambda, double *Qu, int Y_rows, int Y_cols, int W_rows, int W_cols)
{
	//std::cout<<"Inside x \n";
	//find diag(Wu) --> Wu_diag       -----------------------------------------------------------------------------------
	
	double * Wu_diag;
	Wu_diag = (double *)malloc(W_cols * W_cols * sizeof(double));
	
	create_diagonal_matrix(Wu_diag, Wu, W_cols);
	
			/* printf("find diag(Wu) --> Wu_diag  is \n");
			for(int i=0; i<(W_cols*W_cols); i++)
			{
					printf("%f \t", Wu_diag[i]);
			}
			printf("\n");  */
    
	//find Y.T --> Y_trans	---------------------------------------------------------------------------------------------
    
    double *Y_trans;
	Y_trans = (double *)malloc(Y_rows * Y_cols * sizeof(double));
    
	calling_trans(Y_rows, Y_cols, Y, Y_trans);
	
			/* printf("find Y.T --> Y_trans  is \n");
			for(int i = 0; i<(Y_rows * Y_cols); i++){
				printf("%f \t", Y_trans[i]);
			}  
			printf("\n");  */
	
	
	//find eye(n_factors) and multiply with lambda --> eye_lambda_matrix      ------------------------------------------------
	
	double *eye_lambda;
	eye_lambda = (double *)malloc(n_factors * sizeof(double));
	for(int i = 0; i< n_factors; i++){
		eye_lambda[i] = lambda;
	}
	
	double * eye_lambda_matrix;
	eye_lambda_matrix = (double *)malloc(n_factors * n_factors * sizeof(double));
	
	create_diagonal_matrix(eye_lambda_matrix, eye_lambda, n_factors);
	
			/* printf("find eye(n_factors) and multiply with lambda --> eye_lambda_matrix is \n");
			for(int i = 0; i<(n_factors*n_factors); i++){
				printf("%f \t", eye_lambda_matrix[i]);
			} 
			printf("\n");  */
	
	//dot product of Wu_diag, Y_trans --> temp_1      ----------------------------------------------------------------------------
	
	int nr_rows_finalproduct = Y_rows;
    int nr_cols_finalproduct = W_cols;
	
	double *temp_1;
	temp_1 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(W_cols, W_cols, Y_cols, Y_rows, nr_rows_finalproduct, nr_cols_finalproduct, Wu_diag, Y_trans, temp_1);
	
			/* printf("The dot product of Wu_diag, Y_trans --> temp_1 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", temp_1[i]);
			}
			printf("\n");  */
	
	//clear Y_trans
	
	free(Y_trans);   // clearing CPU memory
	
	//dot product of Y, temp_1 --> temp_2		-----------------------------------------------------------------------------------
	
	nr_rows_finalproduct = Y_rows;
    nr_cols_finalproduct = Y_rows;
	
	double *temp_2;
	temp_2 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(Y_rows, Y_cols, W_cols, Y_rows, nr_rows_finalproduct, nr_cols_finalproduct, Y, temp_1, temp_2);
	
			/* printf("dot product of Y, temp_1 --> temp_2 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", temp_2[i]);
			}
			printf("\n");  */ 
	
	//clear temp_1
	
	free(temp_1);   // clearing CPU memory
	
	//addition of temp_2, eye_lambda_matrix --> A (alias add_result)   ----------------------------------------------------------------------------
	
	const int ARRAY_BYTES =  nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double);
	double * A;
	A = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	double * dX_input;
    double * dY_input;
	double * D_A;

	cudaMalloc((void**) &dX_input, ARRAY_BYTES);
	cudaMalloc((void**) &dY_input, ARRAY_BYTES);
    cudaMalloc((void**) &D_A, ARRAY_BYTES);

	cudaMemcpy(dX_input, temp_2, ARRAY_BYTES, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_input, eye_lambda_matrix, ARRAY_BYTES, cudaMemcpyHostToDevice);

	Add<<<1, (nr_rows_finalproduct * nr_cols_finalproduct)>>>(D_A, dX_input, dY_input);
	
	cudaMemcpy(A, D_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
			/* printf("addition of temp_2, eye_lambda_matrix --> A (alias add_result) is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", A[i]);
			}
			printf("\n");  */
	
	//clear temp_2 and eye_lambda_matrix
	
	free(temp_2);   // clearing CPU memory
	free(eye_lambda);
	free(eye_lambda_matrix);
	
	//-------------------------------------
	
	//find transpose of Qu --> Qu_trans      ----------------------------------------------------------------------------
	
	double *Qu_trans;
	Qu_trans = (double *)malloc(1 * W_cols * sizeof(double));
    
	calling_trans(1, W_cols, Qu, Qu_trans);     // is it necessary doing this ?????
	
			/* printf("find transpose of Qu --> Qu_trans is \n");
			for(int i = 0; i < W_cols; i++){
				printf("%f \t", Qu_trans[i]);
			}  
			printf("\n"); */ 
	
	//find dot product Wu_diag, Qu_trans --> temp3		----------------------------------------------------------------------------
	
	nr_rows_finalproduct = 1;
    nr_cols_finalproduct = W_cols;
	
	double *temp3;
	temp3 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(W_cols, W_cols, W_cols, 1, nr_rows_finalproduct, nr_cols_finalproduct, Wu_diag, Qu_trans, temp3);
	
			/* printf("find dot product Wu_diag, Qu_trans --> temp3 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", temp3[i]);
			}
			printf("\n"); */  
	
	//clear Wu_diag and Qu_trans
	
	free(Wu_diag);
	free(Qu_trans);
	
	//dot product of Y(3*5) and temp3(5*1) --> B (3*1)		----------------------------------------------------------------------------
	
	nr_rows_finalproduct = 1;
    nr_cols_finalproduct = Y_rows;
	
	double *B;
	B = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(Y_rows, Y_cols, W_cols, 1, nr_rows_finalproduct, nr_cols_finalproduct, Y, temp3, B);

			/* printf("dot product of Y(3*5) and temp3(5*1) --> B (3*1) is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", B[i]);
			}
			printf("\n");  */

    	//clear temp3
	free(temp3);
	
	//linalg(A, B) --> x_X	-------------------------------------------------------------------------------------------------------------------
	
				/* double *A_final;
				A_final = (double *)malloc(Y_rows*Y_rows*sizeof(double));
				std::copy(A, A + (Y_rows*Y_rows), A_final);

				double *B_final;
				B_final = (double *)malloc(Y_rows*1*sizeof(double));
				std::copy(B, B + (Y_rows*1), B_final); */

	double * x_X;
	x_X = (double *)malloc(Y_rows*1*sizeof(double));
	
	const int nrhs = 1;
	lin_alg_solve(x_X, A, B, Y_rows, nrhs);

				/* double * X_final;
				X_final = (double *)malloc(Y_rows*1*sizeof(double));
				std::copy(x_X, x_X + (Y_rows*1), X_final); */
	
			/* printf("linalg(A, B) --> x_X is \n");
			for(int i = 0; i < (Y_rows*1); i++)
				printf("%f \n",x_X[i]);    */

    	
	// clear A, B
	
	free(A);
	free(B);
	
	for(int i = 0; i < n_factors; i++)
	{
		X[u*n_factors + i] = x_X[i];
	}
	free(x_X);
	
}

void als_solve_Y(double * YT, int i_Y, double * X, double * Wi, int n_factors, double lambda, double * Qi, int X_rows, int X_cols, int W_rows, int W_cols)
{
	// create diagonal matrix of Wi --> Wi_diag    ------------------------------------------------------------------------------------------
	
	double * Wi_diag;
	Wi_diag = (double *)malloc(W_rows * W_rows * sizeof(double));
	
	create_diagonal_matrix(Wi_diag, Wi, W_rows);
	
			/*  printf("find diag(Wi) --> Wi_diag  is \n");
			for(int i=0; i<(W_rows*W_rows); i++)
			{
					printf("%f \t", Wi_diag[i]);
			}
			printf("\n");   */
	
	//dot product of Wi_diag and X --> matrix1     ------------------------------------------------------------------------------------------
	
	int nr_rows_finalproduct = X_cols;
    int nr_cols_finalproduct = W_rows;
	
	double *matrix1;
	matrix1 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(W_rows, W_rows, X_rows, X_cols, nr_rows_finalproduct, nr_cols_finalproduct, Wi_diag, X, matrix1);
	
			/*  printf("The dot product of Wi_diag and X --> matrix1 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", matrix1[i]);
			}
			printf("\n");  */  
	
	//tranpose of X --> X_trans		---------------------------------------------------------------------------------------------------------
	
	double *X_trans;
	X_trans = (double *)malloc(X_cols * X_rows * sizeof(double));
    
	calling_trans(X_rows, X_cols, X, X_trans);
	
			/* printf("find X.T --> X_trans  is \n");
			for(int i = 0; i<(X_cols * X_rows); i++){
				printf("%f \t", X_trans[i]);
			}  
			printf("\n");  */
	
	//dot product of X_trans and matrix1 --> matrix2      ------------------------------------------------------------------------------------------
	
	nr_rows_finalproduct = X_cols;
    nr_cols_finalproduct = X_cols;
	
	double *matrix2;
	matrix2 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(X_cols, X_rows, W_rows, X_cols, nr_rows_finalproduct, nr_cols_finalproduct, X_trans, matrix1, matrix2);
	
			/*  printf("The dot product of X_trans and matrix1 --> matrix2 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", matrix2[i]);
			}
			printf("\n"); */   
	
	free(matrix1);

	//find eye(n_factors) and multiply with lambda --> eye_lambda (matrix)		--------------------------------------------------------------------------------
			
	double *eye_lambda;
	eye_lambda = (double *)malloc(n_factors * sizeof(double));
	for(int i = 0; i< n_factors; i++){
		eye_lambda[i] = lambda;
	}
	
	double * eye_lambda_matrix;
	eye_lambda_matrix = (double *)malloc(n_factors * n_factors * sizeof(double));
	
	create_diagonal_matrix(eye_lambda_matrix, eye_lambda, n_factors);
	
			/* printf("find eye(n_factors) and multiply with lambda --> eye_lambda_matrix is \n");
			for(int i = 0; i<(n_factors*n_factors); i++){
				printf("%f \t", eye_lambda_matrix[i]);
			} 
			printf("\n");   */
	
	free(eye_lambda);	

	//addition of matrix2 and eye_lambda_matrix --> A		----------------------------------------------------------------------------------------------------
	
	const int ARRAY_BYTES =  nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double);
	double * A;
	A = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	double * dX_input;
    	double * dY_input;
	double * D_A;

	cudaMalloc((void**) &dX_input, ARRAY_BYTES);
	cudaMalloc((void**) &dY_input, ARRAY_BYTES);
    cudaMalloc((void**) &D_A, ARRAY_BYTES);

	cudaMemcpy(dX_input, matrix2, ARRAY_BYTES, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_input, eye_lambda_matrix, ARRAY_BYTES, cudaMemcpyHostToDevice);

	Add<<<1, (nr_rows_finalproduct * nr_cols_finalproduct)>>>(D_A, dX_input, dY_input);
	
	cudaMemcpy(A, D_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
			/*  printf("addition of matrix2, eye_lambda --> A is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", A[i]);
			}
			printf("\n");    */
	free(eye_lambda_matrix);
	cudaFree(dX_input);
	cudaFree(dY_input);
	cudaFree(D_A);
	free(matrix2);

	//------------------------------------------
	
	//dot product of Wi_diag and Qi --> matrix3			------------------------------------------------------------------------------------------
	
	nr_rows_finalproduct = 1;
    nr_cols_finalproduct = W_rows;
	
	double *matrix3;
	matrix3 = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(W_rows, W_rows, W_rows, 1, nr_rows_finalproduct, nr_cols_finalproduct, Wi_diag, Qi, matrix3);
	
			/*  printf("The dot product of Wi_diag and Qi --> matrix3 is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", matrix3[i]);
			}
			printf("\n");   */ 

	free(Wi_diag);
	
	//dot product of X_trans and matrix3 --> B		------------------------------------------------------------------------------------------
	
	nr_rows_finalproduct = 1;
    nr_cols_finalproduct = X_cols;
	
	double *B;
	B = (double *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(double));
	
	mul(X_cols, X_rows, W_rows, 1, nr_rows_finalproduct, nr_cols_finalproduct, X_trans, matrix3, B);
	
			/*  printf("The dot product of X_trans and matrix3 --> B is \n");
			for (int i = 0; i<(nr_rows_finalproduct * nr_cols_finalproduct); i++)
			{
				printf("%f \t", B[i]);
			}
			printf("\n");  */ 
	
	free(X_trans);
	free(matrix3);
	
	//linalg(A, B) --> x_Y    -----------------------------------------------------------------------------------------------------------
	
	double * x_Y;
	x_Y = (double *)malloc(X_cols*1*sizeof(double));
	
	const int nrhs = 1;
	lin_alg_solve(x_Y, A, B, X_cols, nrhs);
	
			/*  printf("linalg(A, B) --> x_Y is \n");
			for(int i = 0; i < (X_cols*1); i++)
				printf("%f \n",x_Y[i]); */
			
	for(int i = 0; i < n_factors; i++)
	{
		YT[i_Y*n_factors + i] = x_Y[i];
	}
	
	free(A);
	free(B);
	free(x_Y);	
	
}

__global__ void Subtraction(double * dX_out, double * dX_in, double * dY_in)
{
	int idx = threadIdx.x;
	dX_out[idx] = dX_in[idx] - dY_in[idx];
}

__global__ void element_multiplication(double * dX_out, double * dX_in, double * dY_in)
{
	int idx = threadIdx.x;
	dX_out[idx] = dX_in[idx] * dY_in[idx];
}

void rmse(double * squares, double * Q, double * X, double * Y, double * W, int Q_rows, int Q_cols, int n_factors)
{
	//dot product of X and Y --> Q_hat
	int Q_hat_rows = Q_cols; 
    int Q_hat_cols = Q_rows;
	
	double *Q_hat;
	Q_hat = (double *)malloc(Q_hat_rows * Q_hat_cols * sizeof(double));
	
	mul(Q_rows, n_factors, n_factors, Q_cols, Q_hat_rows, Q_hat_cols, X, Y, Q_hat);
	
		    /*  printf("The dot product of X and Y --> Q_hat is \n");
			for (int i = 0; i<(Q_hat_rows * Q_hat_cols); i++)
			{
				printf("%f \t", Q_hat[i]);
			}
			printf("\n"); */
			
	//subtraction of Q_hat from Q --> sub

	const int ARRAY_BYTES =  Q_rows * Q_cols * sizeof(double);
	double * sub;
	sub = (double *)malloc(Q_rows * Q_cols * sizeof(double));
	double * dX_input;
    double * dY_input;
	double * D_sub;

	cudaMalloc((void**) &dX_input, ARRAY_BYTES);
	cudaMalloc((void**) &dY_input, ARRAY_BYTES);
    cudaMalloc((void**) &D_sub, ARRAY_BYTES);

	cudaMemcpy(dX_input, Q, ARRAY_BYTES, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_input, Q_hat, ARRAY_BYTES, cudaMemcpyHostToDevice);

	Subtraction<<<1, (Q_rows * Q_cols)>>>(D_sub, dX_input, dY_input);
	
	cudaMemcpy(sub, D_sub, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	cudaFree(dX_input);
	cudaFree(dY_input);
	cudaFree(D_sub);
	
			 /*   printf("subtraction of Q_hat from Q --> sub is \n");
			for (int i = 0; i<(Q_rows * Q_cols); i++)
			{
				printf("%f \t", sub[i]);
			}
			printf("\n");  */   
			
	// element by element multiplication of W and sub --> mul_result
	
			//const int ARRAY_BYTES =  Q_rows * Q_cols * sizeof(double);
	double * mul_result;
	mul_result = (double *)malloc(Q_rows * Q_cols * sizeof(double));
	
	double * D_mul;

	cudaMalloc((void**) &dX_input, ARRAY_BYTES);
	cudaMalloc((void**) &dY_input, ARRAY_BYTES);
    cudaMalloc((void**) &D_mul, ARRAY_BYTES);

	cudaMemcpy(dX_input, W, ARRAY_BYTES, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_input, sub, ARRAY_BYTES, cudaMemcpyHostToDevice);

	element_multiplication<<<1, (Q_rows * Q_cols)>>>(D_mul, dX_input, dY_input);
	
	cudaMemcpy(mul_result, D_mul, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	cudaFree(dX_input);
	cudaFree(dY_input);
	cudaFree(D_mul);
	
			/*     printf("element by element multiplication of W and sub --> mul_result is \n");
			for (int i = 0; i<(Q_rows * Q_cols); i++)
			{
				printf("%f \t", mul_result[i]);
			}
			printf("\n");  */ 
			
	// element by element multiplication of mul_result and mul_result --> squares
	
	
	
	double * D_SoS;

	cudaMalloc((void**) &dX_input, ARRAY_BYTES);
	cudaMalloc((void**) &dY_input, ARRAY_BYTES);
    cudaMalloc((void**) &D_SoS, ARRAY_BYTES);

	cudaMemcpy(dX_input, mul_result, ARRAY_BYTES, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_input, mul_result, ARRAY_BYTES, cudaMemcpyHostToDevice);

	element_multiplication<<<1, (Q_rows * Q_cols)>>>(D_SoS, dX_input, dY_input);
	
	cudaMemcpy(squares, D_SoS, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	cudaFree(dX_input);
	cudaFree(dY_input);
	cudaFree(D_SoS);
	
			/*    printf("element by element multiplication of mul_result and mul_result --> squares is \n");
			for (int i = 0; i<(Q_rows * Q_cols); i++)
			{
				printf("%f \t", squares[i]);
			}
			printf("\n");   */
			
	
}

int main()
{	
	int Q_rows = 4;
	int Q_cols = 5;
	double Q_temp[20] = {3, 4, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 3, 4, 0, 0, 0, 0, 1, 3};
	
	double *Q;
	Q = (double *)malloc(Q_rows * Q_cols * sizeof(double));
	memcpy(Q, Q_temp, sizeof(double)*20);
	   
	int n_factors = 3;
	double lambda = 0.1;
	//int n_iterations = 20;
	
	int X_rows = 4;
	int X_cols = n_factors;
	
	double X_temp[12] = {1.04537429,  3.45278132,  3.47493422, 2.24801288,  4.88137731,  1.66288503, 4.81317032,  4.63570752,  1.36892613, 3.32203655,  3.31923711,  2.27048096}; 
	
	double *X;
	X = (double *)malloc(X_rows * X_cols * sizeof(double));
	memcpy(X, X_temp, sizeof(double)*12);
	
	//for(int i = 0; i<12;i++)
	//	printf("%.8f \t", X[i]);
	
	printf("\n");
	int Y_rows = n_factors;
	int Y_cols = 5;
	
	double Y_temp[15] = {1.59982314,  4.78360092,  3.45781337,  3.13286951,  0.50542705, 3.83681956,  2.88250821,  1.1667597 ,  2.43170423,  4.06026517, 0.65686686,  2.94705632,  0.46822364,  1.98082364,  1.54905706};
	
	double *Y;
	Y = (double *)malloc(Y_rows * Y_cols * sizeof(double));
	memcpy(Y, Y_temp, sizeof(double)*15);
	
	int W_rows = Q_rows;
	int W_cols = Q_cols;
	
	double W_temp[20] = {1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1};
	
	double *W;
	W = (double *)malloc(W_rows * W_cols * sizeof(double));
	memcpy(W, W_temp, sizeof(double)*20);
	
	/* for (int i = 0; i<20; i++)
    	{
        	printf("%f \t", W[i]);
    	}
    	printf("\n"); 
	*/
	
	//Main computation starts here with n_iterations
	
	for(int it = 0; it<20;it++)
	{
	
	//updating all rows of X starts here
	
			double *Wu;
			Wu = (double *)malloc(W_cols * sizeof(double));
			//Wu[0] = 1; Wu[1] = 1; Wu[2] = 0; Wu[3] = 1; Wu[4] = 0;
			
			double *Qu;
			Qu = (double *)malloc(W_cols * sizeof(double));
			//Qu[0] = 3; Qu[1] = 4; Qu[2] = 0; Qu[3] = 2; Qu[4] = 0;
	
	//#pragma omp parallel for private(Wu, Qu)
	for(int u = 0; u < W_rows; u++)
	{	
		for(int k = 0; k < W_cols; k++)
		{
			Wu[k] = W[u*W_cols + k];
			Qu[k] = Q[u*W_cols + k];
		} 
		
		als_solve_X(X, u, Y, Wu, n_factors, lambda, Qu, Y_rows, Y_cols, W_rows, W_cols);
	}
		
		/* printf("X with updated row is \n");
		for (int i = 0; i< 12; i++)
			{
				printf("%f \t", X[i]);
			}
			printf("\n"); */
	
	//updating all columns of Y starts here
	
			double *Wi;
			Wi = (double *)malloc(W_rows * sizeof(double));
			//Wi[0] = 1; Wi[1] = 0; Wi[2] = 1; Wi[3] = 0;
			
			double *Qi;
			Qi = (double *)malloc(W_rows * sizeof(double));
			//Qi[0] = 3; Qi[1] = 0; Qi[2] = 1; Qi[3] = 0;
	
	//take the transpose of W to get WT
	
	double *WT;
	WT = (double *)malloc(W_cols * W_rows * sizeof(double));
    
	calling_trans(W_rows, W_cols, W, WT);
	
			/*  printf("find W.T --> WT  is \n");
			for(int i = 0; i<(W_cols * W_rows); i++){
				printf("%f \t", WT[i]);
			}  
			printf("\n");   */
	
	//take the transpose of Q to get QT
	
	double *QT;
	QT = (double *)malloc(Q_cols * Q_rows * sizeof(double));
    
	calling_trans(Q_rows, Q_cols, Q, QT);
	
	//take the transpose of Y to get YT
	
	double *YT;
	YT = (double *)malloc(Y_cols * Y_rows * sizeof(double));
    
	calling_trans(Y_rows, Y_cols, Y, YT);
	
	
	//#pragma omp parallel for private(Wi, Qi)
	for(int i_Y = 0; i_Y < W_cols; i_Y++)
	{	
		for(int p = 0; p < W_rows; p++)
		{
			Wi[p] = WT[i_Y*W_rows + p];
			Qi[p] = QT[i_Y*W_rows + p];
		} 
		
		als_solve_Y(YT, i_Y, X, Wi, n_factors, lambda, Qi, X_rows, X_cols, W_rows, W_cols);
	}
	
	calling_trans(Y_cols, Y_rows, YT, Y);
	
	//call rmse() function to get rmse values
	double * Squares;
	Squares = (double *)malloc(Q_rows * Q_cols * sizeof(double));
	rmse(Squares, Q, X, Y, W, Q_rows, Q_cols, n_factors);
	
	double error = 0;
	
	for(int i = 0; i < (Q_rows * Q_cols); i++)
	{
		error = error + Squares[i];
	}
	printf("%f \n", error);
	
	  }    //n-iterations end here
	
	/* printf("X with updated row is \n");
		for (int i = 0; i< 12; i++)
			{
				printf("%f \t", X[i]);
			}
			printf("\n");
	printf("Y with updated column is \n");
		for (int i = 0; i< 15; i++)
			{
				printf("%f \t", Y[i]);
			}
			printf("\n");  */ 
	
	//dot product of X and Y starts here AFTER the end of n_iterations --> Q_hat
	
	int Q_hat_rows = Q_cols; 
    int Q_hat_cols = Q_rows;
	
	double *Q_hat;
	Q_hat = (double *)malloc(Q_hat_rows * Q_hat_cols * sizeof(double));
	
	mul(Q_rows, Y_rows, Y_rows, Q_cols, Q_hat_rows, Q_hat_cols, X, Y, Q_hat);
	
			  printf("The dot product of X and Y --> Q_hat is \n");
			for (int i = 0; i<(Q_hat_rows * Q_hat_cols); i++)
			{
				printf("%f \t", Q_hat[i]);
			}
			printf("\n");  
			
			printf("Q is \n");
			for (int i = 0; i<(Q_rows * Q_cols); i++)
			{
				printf("%f \t", Q[i]);
			}
			printf("\n");  
	
	// printing the recommendations 
	
	 for(int i = 0; i < Q_rows; i++)
	{
		for(int j = 0; j < Q_cols; j++)
		{
			if((Q[i * Q_cols + j] == 0) && (Q_hat[i * Q_cols + j] > 1.5))
			{
				printf("user %d may like movie %d \n", i, j);
			}
		}
	} 
	
}
