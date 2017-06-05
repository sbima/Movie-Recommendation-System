#include <stdio.h>

void create_diagonal_matrix(float *Dmatrix, float matrix[3], int array_length)
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


int main()
{
	float matrix[3] = {1,2,3};
	int array_length = sizeof(matrix)/sizeof(matrix[0]);
	
	float *Dmatrix;
	Dmatrix = (float *)malloc(array_length*array_length);
	
	create_diagonal_matrix(Dmatrix, matrix, array_length);
	
	for(int i=0; i<(array_length*array_length); i++)
	{
			printf("%f \t", Dmatrix[i]);
	}
	printf("\n");
	return 0;
}
