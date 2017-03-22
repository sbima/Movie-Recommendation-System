#include <stdio.h>

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
__device__ void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) 
{

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

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}

__global__ void update_X(float * dX_out, float * dX_in, float * dY_in)
{
	int idx = threadIdx.x;
	dX_out[idx] = dX_in[idx] + dY_in[idx];
}

__global__ void update_Y(float * dY_out, float * dX_in, float * dY_in)
{
        //int idx = threadIdx.x;
        //dY_out[idx] = dX_in[idx] + dY_in[idx];
}

int main() 
{
        const int ARRAY_BYTES_x = 12 * sizeof(float);
	const int ARRAY_BYTES_y = 12 * sizeof(float);

	// generate the input array on the host
//	float Q[20] = {3,4,0,2,0,0,2,1,0,0,1,0,3,4,0,0,0,0,1,3};
	//float Q_hat[20];
	float h_out[12];
	
	//float test = Q[0*5+3];
	//printf("%f \n", test);

/*	float W[20];
	for (int i = 0; i< 20; i++)
	{
		if(Q[i]>0.5)
		{
			W[i] = 1;
		}
		else
		{
			W[i] = 0;	
		}
	}	
*/
	/*
	for (int i = 0; i<20; i++)
	{
		printf("%f \t", W[i]);
	}
	printf("\n");
	*/

	//float lambda_ = 0.1;
	//int n_factors = 3;
	////m, n = Q.shape
	//int n_iterations = 20;

	float X[12] = {1.04537429, 3.45278132, 3.47493422, 2.24801288, 4.88137731, 1.66288503, 4.81317032, 4.63570752, 1.36892613, 3.32203655, 3.31923711, 2.27048096};
	float Y[12] = {1.59982314, 4.78360092, 3.45781337, 3.13286951, 0.50542705, 3.83681956, 2.88250821, 1.1667597, 2.43170423, 4.06026517, 0.65686686, 2.94705632};
// 0.46822364, 
		//	1.98082364, 1.54905706};

        // declare GPU memory pointers
	float * dX_in;
        float * dY_in;
	float * dX_out;

        // allocate GPU memory
	cudaMalloc((void**) &dX_in, ARRAY_BYTES_x);
	cudaMalloc((void**) &dY_in, ARRAY_BYTES_y);
        cudaMalloc((void**) &dX_out, ARRAY_BYTES_x);

        // transfer the array to the GPU
	cudaMemcpy(dX_in, X, ARRAY_BYTES_x, cudaMemcpyHostToDevice);        
	cudaMemcpy(dY_in, Y, ARRAY_BYTES_y, cudaMemcpyHostToDevice);

        // launch the kernel
        update_X<<<1, 12>>>(dX_out, dX_in, dY_in);

        // copy back the result array to the CPU
        cudaMemcpy(h_out, dX_out, ARRAY_BYTES_x, cudaMemcpyDeviceToHost);

        // print out the resulting array
        for (int i = 0; i<12; i++)
        {
                printf("%f \t", h_out[i]);
        }
        printf("\n");

        cudaFree(dX_in);
	cudaFree(dY_in);
        cudaFree(dX_out);

        return 0;
}
