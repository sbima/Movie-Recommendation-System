# Movie-Recommendation-System
Depending upon the ratings given to the multiple movies by the multiple users, the system can recommend some movies to the user. This design is to be done using CUDA programming on GPU.

I am using c based CUDA programming to parallelize the ALS(Alternating Least Squares) algorithm on GPU. ALS is the algorithm by using which the system decides what movies are to be recommended to a particular user.

# Parallelizing Alternating Least Squares Algorithm on CUDA runtime Environment
Project Description:
	Parallelize the sequential ALS algorithm using c based CUDA and analyze the performance results of both sequential and parallel programs.

# Implementation:
•	Used cuBLAS API: gpublasDgemm
 	To calculate the dot products of any intermediate results within the program in parallel on GPU Device.
•	Used cuSOLVER Library: 
This library does the QR decomposition to solve the linear system, which is Ax = B. Where A is a dense matrix with equal number of rows and columns.Main computations needed for this decomposition are done in parallel on Device.
The code uses three steps:
Step 1: A = Q*R by geqrf.
Step 2: B := Q^T*B by ormqr.
Step 3: solve R*X = B by trsm.
Finally, x = R \ Q^T*B
•	I have written a kernel named “trans” to find the Transpose of any 1-Dimensional matrix whenever necessary. 
•	Trying to use OpenMP: 
I tried to parallelize the main part of the program where I can do large amount of computations using OpenMP.
# Data: Input to the program
 
An input matrix is developed with the ratings given by a set of users to a set of users. This data is collected from ratings.txt file.
1)	user-id as row-index    		2) movie-id as column-index


![alt text](http://webspace.cs.odu.edu/~sbimavar/Picture1.png "Weighted errors")

![alt text](http://webspace.cs.odu.edu/~sbimavar/Picture2.png "errors")
