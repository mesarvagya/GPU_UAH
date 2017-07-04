// gpu (device) based matrix/matrix gpu code
//-------------------------------------------------------------------------
// Included CUDA libraries
//-------------------------------------------------------------------------
#include <stdio.h>

// iceil macro
// returns an integer ceil value where integer numerator is first parameter
// and integer denominator is the second parameter. iceil is the rounded
// up value of numerator/denominator when there is a remainder
// equivalent to ((num%den!=0) ? num/den+1 : num/den)
#define iceil(num,den) (num+den-1)/den 

#define TILE_WIDTH 16 // block x and y dimensions

void check_error(cudaError_t error_id){
   if (error_id != cudaSuccess) {
      printf("Error is %d", error_id);
      exit(EXIT_FAILURE);
   }
}

// GPU device MatrixMulKernel kernel code 
__global__ void MatrixMulKernel(float *Pd, float *Md, float *Nd, int Mh,
   int Mw, int Nw) {
   // ==================================================================
   // Solution part 4
   // Determine the output index of each thread.
   // Compute the dot product of one row of Md and one column of Nd
   // for each thread.
   // Write the computed value to matrix P at the correct output index
   // ==================================================================

   // Calculate the global row and column indices of the Pd matrix
   int Row;
   int Col;
   //**** ENTER YOUR CODE HERE ****
   Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
   Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
   if(Row <Mh && Col< Nw){

   // Each thread computes one dot product element of the block sub-matrix
   // access correct row of Md and Column of Nd assuming row-major allocations
   // (Note: in second part of hw1 you will want to make sure that only
   //  the threads that are assigned valid regions of the computation are
   //  active
   //**** ENTER YOUR CODE HERE ****

   float Pvalue = 0;
   //**** ENTER YOUR CODE HERE ****
   for(int k=0; k < Mw ;++k){
      Pvalue += Md[Row*Mw+k] * Nd[k*Nw + Col];
   }

   // place final result in specified location of global Pd memory
   //**** ENTER YOUR CODE HERE ****
   Pd[Row * Nw + Col] = Pvalue;
   }

   // End of solution part 4 ===========================================
}

__global__ void MatrixMulKernelSingleBlock(float *Pd, float *Md, float *Nd, int Mh,
   int Mw, int Nw) {
   // ==================================================================
   // Solution part 4
   // Determine the output index of each thread.
   // Compute the dot product of one row of Md and one column of Nd
   // for each thread.
   // Write the computed value to matrix P at the correct output index
   // ==================================================================

   // Calculate the global row and column indices of the Pd matrix
   int Row;
   int Col;
   //**** ENTER YOUR CODE HERE ****
   Row = threadIdx.y;
   Col = threadIdx.x;
   if(Row > Mh || Col > Nw) return;

   // Each thread computes one dot product element of the block sub-matrix
   // access correct row of Md and Column of Nd assuming row-major allocations
   // (Note: in second part of hw1 you will want to make sure that only
   //  the threads that are assigned valid regions of the computation are
   //  active
   //**** ENTER YOUR CODE HERE ****

   float Pvalue = 0;
   //**** ENTER YOUR CODE HERE ****
   for(int k=0; k < Mw ;++k){
      Pvalue += Md[Row*Mw+k] * Nd[k*Nw + Col];
   }

   // place final result in specified location of global Pd memory
   //**** ENTER YOUR CODE HERE ****
   Pd[Row * Nw + Col] = Pvalue;

   // End of solution part 4 ===========================================
}

void compute_GPU(float *P, float *M, float *N, int Mh, 
   int Mw, int Nw) {
   float *Md, *Nd, *Pd;
   cudaError_t error_id;

   // ===================================================================
   // Solution part 1: Copy Input Data from Host to Device
   //    Create Device Buffers for the two input matrices
   //    Copy memory from the host memory to the device buffer (device memory)
   //    Check for error generated while using each OpenCL API call
   // ===================================================================


   // Allocate device memory and Transfer host arrays M and N 
   //**** ENTER YOUR CODE HERE ****
   size_t size_M = Mh * Mw * sizeof(float);
   size_t size_N =  Mw*Nw * sizeof(float);
   error_id = cudaMalloc((void**)&Md, size_M);
   check_error(error_id);
   error_id = cudaMemcpy(Md, M, size_M, cudaMemcpyHostToDevice);
   check_error(error_id);

   error_id = cudaMalloc((void**)&Nd,size_N);
   check_error(error_id);
   error_id = cudaMemcpy(Nd, N, size_N, cudaMemcpyHostToDevice);
   check_error(error_id);

   // Allocate device memory of P array for results
   //**** ENTER YOUR CODE HERE ****
   size_t size_P = Mh*Nw*sizeof(float);
   error_id = cudaMalloc((void**)&Pd, size_P );
   check_error(error_id);

   // End of solution Part 1 ============================================


   // ===================================================================
   // Solution part 2
   //    A. Initialize the block and grid dimensions of the kernel about
   //       to be launched.
   //       [You may assume that each matrix dimension is a multiple of the
   //        defined constant block_size.]
   //    B. Launch the kernel with appropriate kernel arguments
   //    Do not forget to check for success at each stage before proceeding.
   // ===================================================================

   // Setup the kernel execution configuration parameters/launch kernel

   // Stage A:  Setup the kernel execution configuration parameters
   //           (in second part of homework take into account the case where
   //            the dimmensions are not an even multiple of block size)
   //**** ENTER YOUR CODE HERE ****

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   //         (remember to check for kernel launch failure!)
   //**** ENTER YOUR CODE HERE ****
   if(Mh == 16 && Mw == 16 && Nw == 16){
      // Single Block 16*16 testing
      dim3 grid(1,1);
      dim3 block(TILE_WIDTH, TILE_WIDTH);
      MatrixMulKernelSingleBlock<<<grid,block>>>(Pd, Md, Nd,Mh, Mw, Nw);
   }
   else
   {
      int a1 = iceil(Mh, TILE_WIDTH);
      int a2 = iceil(Nw, TILE_WIDTH);
      dim3 grid(a2,a1);
      dim3 block(TILE_WIDTH, TILE_WIDTH);
      MatrixMulKernel<<<grid,block>>>(Pd, Md, Nd,Mh, Mw, Nw);
   }
   // End of solution Part 2 ============================================


   // ===================================================================
   // Solution part 3
   // Copy Results Device back to Host
   // ===================================================================

   // Transfer P from device to host
   //**** ENTER YOUR CODE HERE ****
   error_id = cudaMemcpy(P,Pd,size_P,cudaMemcpyDeviceToHost);
   check_error(error_id);


   // End of solution Part 3 ============================================


   // CLEAN UP -- Free device memory when finished
   //**** ENTER YOUR CODE HERE ****
   error_id = cudaFree(Md);
   check_error(error_id);
   error_id = cudaFree(Nd);
   check_error(error_id);
   error_id = cudaFree(Pd);
   check_error(error_id);

}
