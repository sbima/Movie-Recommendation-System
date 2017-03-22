#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>
#include <curand.h>

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) 
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

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


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
//            std::cout << A[j * nr_rows_A + i] << " ";
		printf("%f \t", A[j * nr_rows_A + i]);
        }
//        std::cout << std::endl;
		printf("\n");
    }
//    std::cout << std::endl;
	printf("\n");
}

int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A = 2;
	int nr_cols_A = 2;
	int nr_rows_B = 2;
	int nr_cols_B = 3;
	int nr_rows_C = 2;
	int nr_cols_C = 3;

	// for simplicity we are going to use square arrays
	//nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
	
	//float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	//float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	//float h_A[12] = {1.04537429, 3.45278132, 3.47493422, 2.24801288, 4.88137731, 1.66288503, 4.81317032, 4.63570752, 1.36892613, 3.32203655, 3.31923711, 2.27048096};
        //float h_B[15] = {1.59982314, 4.78360092, 3.45781337, 3.13286951, 0.50542705, 3.83681956, 2.88250821, 1.1667597, 2.43170423, 4.06026517, 0.65686686, 2.94705632, 0.46822364,
          //              1.98082364, 1.54905706};	
	
	float h_Ab[4] = {1,2,3,4};
	float h_Bb[6] = {2,3,1,0,4,2};
	float h_A[4];
	float h_B[6];
	int i,j;

	for(i=0; i<nr_rows_A; ++i)
        {
		for(j=0; j<nr_cols_A; ++j)
        	{
            		h_A[j][i] = h_Ab[i][j];
        	}
	}

	nr_rows_A = j;
	nr_cols_A = i;	

	for(i=0; i<nr_rows_B; ++i)
        {
                for(j=0; j<nr_cols_B; ++j)
                {
                        h_B[j][i] = h_Bb[i][j];
                }
        }

	nr_rows_B = j;
	nr_cols_B = i;

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	//GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	//GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
//	std::cout << "A =" << std::endl;
	printf("A = \n");
	print_matrix(h_A, nr_rows_A, nr_cols_A);
//	std::cout << "B =" << std::endl;
	printf("B = \n");
	print_matrix(h_B, nr_rows_B, nr_cols_B);

	// Multiply A and B on GPU
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	std::cout << "C =" << std::endl;
	printf("C = \n");
	print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	

	// Free CPU memory
//	free(h_A);
//	free(h_B);
//	free(h_C);

	return 0;
}

/*
29.756466 	10.994524 	12.037519 	11.485214 	12.279160 	
24.965399 	24.403637 	19.971069 	24.901730 	10.056580 	
40.060867 	26.054535 	23.703743 	27.052744 	16.302774 	
33.622696 	18.097168 	17.409809 	18.863806 	13.752195
*/
