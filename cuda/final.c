#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> 
#include <cusolverDn.h>

void create_diagonal_matrix(float *Dmatrix, float matrix[array_length], int array_length)
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

void als_solve(float *Y, float *Wu, int n_factors, float lambda, float *Qu, int W_rows, int W_cols)
{
	//find diag(Wu) --> Wu_diag
    
	int array_length = sizeof(Wu)/sizeof(Wu[0]);
	
	float *Wu_diag;
	Wu_diag = (float *)malloc(array_length*array_length);
	
	create_diagonal_matrix(Wu_diag, Wu, array_length);
	
	/*for(int i=0; i<(array_length*array_length); i++)
	{
			printf("%f \t", Wu_diag[i]);
	}
	printf("\n");
    */
    
	//find Y.T --> Y_trans
    
    float *Y_trans;
	Y_trans = (float *)malloc(W_rows * W_cols * sizeof(float));
    
	calling_trans(W_rows, W_cols, Y, Y_trans);
	
	//find eye(n_factors) and multiply with lambda --> lambda_matrix
    
    
	
	//dot product of Wu_diag, Y_trans --> temp_1
	//clear Y_trans
	//dot product of Y, temp_1 --> temp_2
	
	//clear temp_1
	
	//addition of temp_2, lambda_matrix --> A
	
	//clear temp_2
	//clear lambda_matrix
	
	----------
	
	//find transpose of Qu --> Qu_trans
	//find dot product Wu_diag, Qu_trans --> temp3
	
	//clear Wu_diag
	//clear Qu_trans
	//dot product of Y and temp3 --> B
	
	//clear temp3
	
	//linalg(A, B) --> X
	
	//clear A, B
	
	//find X transpose --> X_final_row
	
	
}

int main()
{
	int n_factors = 3;
	
	int X_rows = 4;
	int X_cols = n_factors;
	
	float X_temp[12] = {1.04537429,  3.45278132,  3.47493422, 2.24801288,  4.88137731,  1.66288503, 4.81317032,  4.63570752,  1.36892613, 3.32203655,  3.31923711,  2.27048096};
	
	float *X;
	X = (float *)malloc(X_rows * X_cols * sizeof(float));
	memcpy(X, X_temp, sizeof(float)*12);
	
	int Y_rows = n_factors;
	int Y_cols = 5;
	
	float Y_temp[15] = {1.59982314,  4.78360092,  3.45781337,  3.13286951,  0.50542705, 3.83681956,  2.88250821,  1.1667597 ,  2.43170423,  4.06026517, 0.65686686,  2.94705632,  0.46822364,  1.98082364,  1.54905706};
	
	float *Y;
	Y = (float *)malloc(Y_rows * Y_cols * sizeof(float));
	memcpy(Y, Y_temp, sizeof(float)*15);
	
	int W_rows = 4;
	int W_cols = 5;
    
    float *Q;
	Q = (float *)malloc(W_rows * W_cols * sizeof(float));
	memcpy(Q, Q_temp, sizeof(float)*20);
    
    float Q_temp[20] = {3, 4, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 3, 4, 0, 0, 0, 0, 1, 3};
	
	float W_temp[20] = {1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1};
	
	float *W;
	W = (float *)malloc(W_rows * W_cols * sizeof(float));
	memcpy(W, W_temp, sizeof(float)*20);
	
	/* for (int i = 0; i<20; i++)
    	{
        	printf("%f \t", W[i]);
    	}
    	printf("\n"); 
	*/
	
    float Wu_temp[5] = {1, 1, 0, 1, 0};
    float Qu_temp[5] = {3, 4, 0, 2, 0};
	float *Wu;
	Wu = (float *)malloc(W_cols * sizeof(float));
    memcpy(Wu, Wu_temp, sizeof(float)*5);
	
	float *Qu;
	Qu = (float *)malloc(W_cols * sizeof(float));
    memcpy(Qu, Qu_temp, sizeof(float)*5);
	
	/*for(int u = 0; u < X_rows; u++)
    {	
		for(int i = 0; i < W_cols; i++)
        {
			Wu[i] = W[u*W_cols + i];
			Qu[i] = Q[u*W_cols + i];
		}*/
    
        
		
		als_solve(Y, Wu, n_factors, lambda, Qu, Y_rows, Y_cols, W_rows, W_cols)
	//}
}
