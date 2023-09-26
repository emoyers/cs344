//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

#define NUMBER_BITS_IN_BYTE (8u)
#define MIN(X,Y) (X<Y)?X:Y

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void create_histogram_per_digit(const unsigned int* const input_vals, const size_t input_size,
                                           const unsigned int mask, const unsigned int starting_bit, 
                                           unsigned int* const output_hist, const unsigned int output_size)
{
  unsigned int temp_val;
  size_t thread_id = threadIdx.x;

  unsigned int idx = thread_id;
  while(idx < input_size)
  {
    temp_val = MIN((input_vals[idx] >> starting_bit) & mask, output_size-1u);
    atomicAdd(&output_hist[temp_val], 1);

    idx += blockDim.x;
  }

}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  const unsigned int threadsPerBlock = 1024;
  const dim3 blockSize(1, 1, 1);
  const dim3 gridSize( threadsPerBlock, 1, 1);

  const unsigned int mask =  0xffff;
  const unsigned int numBits = 16;
  const unsigned int numBins = 1 << numBits;
  unsigned int* d_binHistogram;
  unsigned int* d_binScan; 

  checkCudaErrors(cudaMalloc(&d_binHistogram, numBins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binScan, numBins*sizeof(unsigned int)));

  for (unsigned int i=0u; i< NUMBER_BITS_IN_BYTE * sizeof(unsigned int); i+= numBits)
  {

    checkCudaErrors(cudaMemset(d_binHistogram, 0, numBins*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_binScan, 0, numBins*sizeof(unsigned int)));

    /* 1) Histogram of the number of occurrences of each digit */
    create_histogram_per_digit<<<blockSize, gridSize>>>(d_inputVals, numElems, 
                                                        mask, i, d_binHistogram, 
                                                        numBins);
    
    /* 2) Exclusive Prefix Sum of Histogram */

  }

  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_binScan));
}
