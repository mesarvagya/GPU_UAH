/**************************************************************************
   File Name [matrixmul.cu]
   Synopsis [This file defines the main function to perform
             matrix-matrix multiplication using cuda.]
   Description[Matrix multiplication: P = M * N]
**************************************************************************/
//-------------------------------------------------------------------------
// Included C libraries
//-------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//-------------------------------------------------------------------------
// Included Portable timer functions 
//-------------------------------------------------------------------------
#include "timer.h"
//-------------------------------------------------------------------------
// Included helper functions 
//-------------------------------------------------------------------------
#include "assist.h"
//-------------------------------------------------------------------------
// Included host and gpu matrix-matrix multiplication function prototypes
#include "matrixmul.h"

/*========================================================================*/
/*                                                                        */
/*  Synopsis  [Main function]                                             */
/*  Description[Matrix multiplication: P = M * N]                         */
/*                                                                        */
/*========================================================================*/

int main (int argc, char **argv) {
   bool if_quiet = false;
   double timer_val;
   int i,j,dev_id=0;
   char *input_fn_M=NULL,*input_fn_N=NULL,*gold_fn=NULL,*gpu_fn=NULL;
   int Mw=0, Mh=0, Nw=0, Nh=0, Pw=0, Ph=0;

   if (argc==2) {
      Mw=Mh=Nw=Nh=Pw=Ph=atoi(argv[1]);
   }
   else {
      if (argc==3) {
         Mw=Mh=Nw=Nh=Pw=Ph=atoi(argv[1]);
         dev_id=atoi(argv[2]);
      }
      else {
         if (argc==4) {
            Ph=Mh=atoi(argv[1]);
            Mw=Nh=atoi(argv[2]);
            Pw=Nw=atoi(argv[3]);
         }
         else {
            if (argc==5) {
               Ph=Mh=atoi(argv[1]);
               Mw=Nh=atoi(argv[2]);
               Pw=Nw=atoi(argv[3]);
               dev_id=atoi(argv[4]);
            }
            else {
               fprintf(stderr,"Error: Wrong number of input parameters.\n");
               fprintf(stderr,"Usage:\n"
                  "$> matrixmul_cu [Mh] <Mw Nw> <device number>]\n"
                  "Examples:\n"
                  "   matrixmul_cu 128\n"
                  "   matrixmul_cu 128 1\n"
                 "   matrixmul_cu 128 128 128\n"
                 "   matrixmul_cu 128 128 128 1\n");
               exit(1);
            }
         }
      }
   }

   // check for GPU device -- output dev info if found
   if (DeviceSelect(dev_id)<0) { 
      fprintf(stderr,"Error: No GPU Device Found\n");
      exit(1);
   }

   // output general information about CUDA Device that is selected
   DeviceInfo(dev_id);

   input_fn_M=(char *) malloc(30*sizeof(char));
   input_fn_N=(char *) malloc(30*sizeof(char));
   gold_fn=(char *) malloc(30*sizeof(char));
   gpu_fn=(char *) malloc(30*sizeof(char));
   sprintf(input_fn_M,"matrix_M_%d.bin",Mh);
   sprintf(input_fn_N,"matrix_N_%d.bin",Mh);
   sprintf(gold_fn,"matrix_%d.gold",Mh);
   sprintf(gpu_fn,"matrix_%d.gpu",Mh);
   if (Pw*Ph>15*15) {
      if_quiet=true; // If not display matrix contents
   }
   printf("Input matrix sizes:\n");
//-------------------------------------------------------------------------
// Setup host side 
//-------------------------------------------------------------------------
   printf("Setup host side environmnet:\n");
   // allocate host memory for matrices M and N  
   printf(" M:%d x %d\n",Mh,Mw); 
   printf(" N:%d x %d\n",Nh,Nw); 
   unsigned int size_M=Mw*Mh;
   unsigned int mem_size_M = sizeof(float)*size_M;
   float *hostM=(float*) malloc(mem_size_M);
   unsigned int size_N=Nw*Nh;
   unsigned int mem_size_N = sizeof(float)*size_N;
   float *hostN=(float*) malloc(mem_size_N);

   // allocate memory for the result on host side
   printf("Allocate memory for the result on host side.\n");
   unsigned int size_P = Pw*Ph;
   unsigned int mem_size_P = sizeof(float)*size_P;
   float *hostP = (float*) malloc(mem_size_P);

   // Initialize the input matrices.
   printf("Generate input matrix data for matrix M and N.\n");
   GenMatrixFile(input_fn_M, Mw, Mh, if_quiet);
   GenMatrixFile(input_fn_N, Nw, Nh, if_quiet);

   unsigned int *matrix=AllocateMatrixSpace(Mw,Mh,Nw);
   ReadMatrixFile(matrix,input_fn_M,Mw,Mh,true);

   for (i=0; i<Mh;i++) {
      for (j=0; j<Mw;j++) {
         hostM[i*Mw+j] = (float) matrix[i*Mw+j];
      }
   }

   ReadMatrixFile(matrix,input_fn_N,Nw,Nh,true);
   for (i=0; i<Nh;i++) {
      for (j=0; j<Nw;j++) {
         hostN[i*Nw+j] = (float) matrix[i*Nw+j];
      }
   }
   //=====================================================================
   // Perform matrix-matrix multiplication on host as a reference
   //=====================================================================
   printf("Computing matrix multiplication MxN on host;\n");
   printf("Output matrix size:\n");
   printf(" P_host:%d x %d\n",Ph,Pw); 
   if (Pw*Ph > 512*512) {
      printf("It takes time since matrix is larger than 512x512.\n");
   }
   float* reference = (float*) malloc(mem_size_P);

   const char *CPU_time={"CPU_time.txt"},*GPU_time={"GPU_time.txt"};

   // Start timer
   startTimer(&timer_val);

   computeGold(reference, hostM, hostN, Mh, Mw, Nw);

   // End timer -- record elapsed time
   timer_val=stopNreadTimer(&timer_val);

   printf(" CPU (Host) Processing time : %f (ms)\n",timer_val*1000.);
   // append run time data to "CPU_time.txt" file
   write_tm(CPU_time, Ph, timer_val);

   printf("Host Computed Reference Matrix data checksum:%g\n",
      CheckSum(reference,Pw, Ph));

   if (!if_quiet) {
      printf(" Host Computed Reference Matrix data contents :\n");
      printf(" ");
   }
   // Convert Host Matrix to unsigned and store in separte memory
   // area. Print results if if_quiet flag is not true
   matrix = (unsigned int*)malloc(Pw*Ph*sizeof(unsigned int));
   for (i = 0; i < Ph; i++) {
      for (j = 0; j < Pw; j++) {
         matrix[i*Pw+j] = (unsigned int)reference[i*Pw+ j];
         if (!if_quiet) printf("%u ", matrix[i*Pw+j]);
      }
      if (!if_quiet) printf("\n ");
   }  
   if (!if_quiet) printf("\n");

   // for some reason save the host unsigned int matrix data in a file
   WriteMatrixFile(gold_fn, matrix, Pw, Ph, 1);

   //=====================================================================
   // Perform matrix-matrix multiplication on gpu device 
   //=====================================================================
   printf("Computing matrix multiplication MxN on GPU;\n");
   printf("Output matrix size:\n");
   printf(" P_gpu:%d x %d\n",Ph,Pw); 

   // Start timer
   startTimer(&timer_val); 

   compute_GPU(hostP, hostM, hostN, Mh, Mw, Nw);

   // End timer -- record elapsed time
   timer_val=stopNreadTimer(&timer_val);

   printf(" GPU (Device) Processing time : %f (ms)\n",timer_val*1000);
   // append run time data to "GPU_time.txt" file
   write_tm(GPU_time, Ph, timer_val); 

   printf("GPU Computed Matrix data checksum:%g\n",
      CheckSum(hostP,Pw, Ph));

   if (!if_quiet) {
      printf(" Matrix data contents :\n");
      printf(" ");
   }
   
   // Convert Device Matrix to unsigned and store in separte memory
   // area. Print results if if_quiet flag is not true
   matrix = (unsigned int*)malloc(Pw*Ph*sizeof(unsigned int));
   for (i = 0; i < Ph; i++) {
      for (j = 0; j < Pw; j++) {
         matrix[i*Pw+j] = (unsigned int)hostP[i*Pw+ j];

         if (!if_quiet) printf("%u ", matrix[i*Pw+j]);
      }
      if (!if_quiet) printf("\n ");
   }  
   if (!if_quiet) printf("\n");

   WriteMatrixFile(gpu_fn, matrix, Pw, Ph, 1);

   CompareMatrixFile (gpu_fn, gold_fn, Pw,Ph, 1);

   // clean up memory on host
   free(matrix); matrix = NULL;
   free(reference);
   free(hostM); free(hostN); free(hostP);
   free(input_fn_N); free(input_fn_M); free(gold_fn); free(gpu_fn);

   return 0;
}


