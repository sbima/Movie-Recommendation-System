#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>
#include <curand.h>

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) 
{
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

__global__ void trans(float * dX_out, float * dX_in, int total_rows, int total_columns)
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

void calling_trans(int nr_rows_A, int nr_cols_A, float * h_Ab, float * h_Ab_out)
{
	const int ARRAY_BYTES_x = nr_rows_A * nr_cols_A * sizeof(float);
	
	float * dX_in;
	float * dX_out;

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

void mul(int nr_rows_A, int nr_cols_A, int nr_rows_B, int nr_cols_B, int nr_rows_finalproduct, int nr_cols_finalproduct, float * h_Ab, float * h_Bb, float * final)
{
	//transpose of first matrix starts here
	float *h_Ab_out;
	h_Ab_out = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	
	calling_trans(nr_rows_A, nr_cols_A, h_Ab, h_Ab_out);
	
	//transpose of second matrix starts here
	float *h_Bb_out;
	h_Bb_out = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	
	calling_trans(nr_rows_B, nr_cols_B, h_Bb, h_Bb_out);
	
	//multiplication to get the resultant matrix starts here
		int nr_rows_product = nr_rows_A;
		int nr_cols_product = nr_cols_B;
	
	float *product = (float *)malloc(nr_rows_product * nr_cols_product * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_product * nr_cols_product * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	cudaMemcpy(d_A,h_Ab_out,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_Bb_out,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_product, nr_cols_A, nr_cols_product);
	
	cudaMemcpy(product,d_C,nr_rows_product * nr_cols_product * sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
	
	// Now, taking the transpose of the resultant matrix (resultant matrix's dimensions (no.of.rows switched with no.of.cols)) gives us the final correct answer 
	
	calling_trans(nr_rows_finalproduct, nr_cols_finalproduct, product, final);
	
}

int main()
{
	//transpose of first matrix starts here
	int nr_rows_A = 2;
	int nr_cols_A = 2;
	
	float *h_Ab;
	h_Ab = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	for(int i=0;i<(nr_rows_A * nr_cols_A);i++)
	{
		h_Ab[i] = i+1;
	}
	
	//transpose of second matrix starts here
	int nr_rows_B = 2;
	int nr_cols_B = 3;
	
	float *h_Bb;
	h_Bb = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	
	h_Bb[0] = 2; h_Bb[1] = 3; h_Bb[2] = 1; h_Bb[3] = 0; h_Bb[4] = 4; h_Bb[5] = 2;
	
	//final product starts here
	int nr_rows_finalproduct = nr_cols_B;
    int nr_cols_finalproduct = nr_rows_A;
	
	float *final;
	final = (float *)malloc(nr_rows_finalproduct * nr_cols_finalproduct * sizeof(float));
	
	mul(nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_finalproduct, nr_cols_finalproduct, h_Ab, h_Bb, final);
	
	printf("The final matrix is \n");
	for (int i = 0; i<6; i++)
	{
		printf("%f \t", final[i]);
	}
	printf("\n");
		
	// Free CPU memory
	//free(h_A);
	//free(h_B);

	return 0;
}

