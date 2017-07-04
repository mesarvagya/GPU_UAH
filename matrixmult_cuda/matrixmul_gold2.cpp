// Alternative "Gold Standard File" that uses a CPU Cache Friendly
// Block decomposition scheme that is analogous to the shared
// memory GPU tiling scheme that will be used in HW3's GPU Optimization
// Exercize (Chapter 6).
// With a TILE WIDTH of 32 (block size) and for large data sizes
// this computes the results from 5 to 8 times faster than
// the row/column method -- Replace the "matrixmul_gold.cpp"
// file with this one.
// B. Earl Wells -- May 25, 2014
/******************************************************
   File Name [matrixmul_gold.cpp]

   Synopsis [This file defines the gold-version matrix- matrix
             multiplication.]

   Description []
*******************************************************/
#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 32 // block size
/*=====================================================*/
/*                                                     */
/* Synopsis [Sequential/Gold version of matrix-matrix  */
/*           multiplication.]                          */
/*i                                                    */
/* Description [This function computes multiplication  */
/* of two matrix M and N, and stores the output to P.] */
/*                                                     */
/*=====================================================*/
void computeGold(float* P, // Resultant matrix data
   const float* M, // Matrix M
   const float* N, // Matrix N
   int Mh, // Matrix M height
   int Mw, // Matrix M width
   int Nw) // Matrix N width
{
   int block_idx_x,block_idx_y,sblk_idx_x;
   int block_dim_y = (Mh+TILE_WIDTH-1)/TILE_WIDTH; // ceil(Mh/TILE_WIDTH)
   int block_dim_x = (Mh+TILE_WIDTH-1)/TILE_WIDTH; // square blocks assumed

   float sum, a, b;
   int i, j, k, Row,Col,k_global;

   for (block_idx_y=0; block_idx_y < block_dim_y;block_idx_y++) {
      for (block_idx_x=0;block_idx_x < block_dim_x;block_idx_x++) {

         // processing a block of the M and N matrix to take advantage
         // of temporal locality of reference that is exploitable in the cache
         for (sblk_idx_x=0;sblk_idx_x < block_dim_x; sblk_idx_x++) {
    
            // for each block compute the partial sums store in targeted location
            // for each block that is being computed
            for (i = 0; i < TILE_WIDTH; i++) {
               Row = TILE_WIDTH*block_idx_y+i;
               if (Row >= Mh) break;     // if out of range do not write
               for (j = 0; j < TILE_WIDTH; j++){
                  Col =  TILE_WIDTH*block_idx_x+j;
                  if (Col >= Nw) break; // if out of range do not write
                  sum = 0;
                  for (k = 0; k < TILE_WIDTH; k++){
                     k_global = TILE_WIDTH*sblk_idx_x+k;
                     if (k_global >= Mw) break; // do not sum out of range values
                     a = M[Row*Mw + k_global];
                     b = N[k_global*Nw+Col];
                     sum += a * b;
                  }
                  if (sblk_idx_x==0)
                     P[Row*Mw + Col]=sum;
                  else
                     P[Row*Mw + Col]+=sum;
               }
            }
         }
      }
   }
}
