/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "float.h"
#include <iostream> // remove this

#define MIN(X,Y) (X<Y)?X:Y
#define MAX(X,Y) (X>Y)?X:Y

__global__ void reduce_min(const float* const d_input, const size_t input_size, 
                           float* const d_output)
{

   extern __shared__ float sh_data[];
   int array_id = threadIdx.x + blockDim.x * blockIdx.x;
   int thread_id = threadIdx.x;

   // load shared mem from global memory
   if(array_id < input_size -1)
   {
      sh_data[thread_id] = d_input[array_id];
   }
   else
   {
      sh_data[thread_id] = FLT_MAX;
   }

   // Do reduction on block level
   for(unsigned int s=blockDim.x/2; s > 0; s>>=1)
   {
      if(thread_id < s)
      {
         sh_data[thread_id] = MIN(sh_data[thread_id],sh_data[thread_id+s]);
      }
      __syncthreads();
   }

   if(thread_id == 0)
   {
      d_output[blockIdx.x] = sh_data[0];
   }

}

__global__ void reduce_max(const float* const d_input, const size_t input_size, 
                           float* const d_output)
{

   extern __shared__ float sh_data[];
   int array_id = threadIdx.x + blockDim.x * blockIdx.x;
   int thread_id = threadIdx.x;

   // load shared mem from global memory
   if(array_id < input_size -1)
   {
      sh_data[thread_id] = d_input[array_id];
   }
   else
   {
      sh_data[thread_id] = FLT_MIN;
   }

   // Do reduction on block level
   for(unsigned int s=blockDim.x/2; s > 0; s>>=1)
   {
      if(thread_id < s)
      {
         sh_data[thread_id] = MAX(sh_data[thread_id],sh_data[thread_id+s]);
      }
      __syncthreads();
   }

   if(thread_id == 0)
   {
      d_output[blockIdx.x] = sh_data[0];
   }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

   /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum */
   
   int threadsPerBlock = 1024;
   int numberBlocks = (int)std::ceil(double(numRows*numCols) / (double)threadsPerBlock);
   const dim3 blockSize(numberBlocks, 1, 1);
   const dim3 gridSize( threadsPerBlock, 1, 1);

   // find next power of 2 for numberBlocks
   int powerTwoBlockSize = 1;
   while(powerTwoBlockSize < numberBlocks)
   {
      powerTwoBlockSize *=2;
   }

   float* d_result_min_max;

   checkCudaErrors(cudaMalloc(&d_result_min_max, powerTwoBlockSize*sizeof(float)));
   
   // First pass reduction, doing only reductions per block
   reduce_min<<<blockSize,gridSize,sizeof(float)*threadsPerBlock>>>(d_logLuminance, numRows*numCols, 
                                                                    d_result_min_max);
   // Doing reduction using the output of the previous step to collect the overall minimum
   reduce_min<<<dim3(1,1,1),dim3(powerTwoBlockSize,1,1),
                sizeof(float)*threadsPerBlock>>>(d_result_min_max, numberBlocks, d_result_min_max);

   checkCudaErrors(cudaMemcpy(&min_logLum, d_result_min_max, sizeof(float), cudaMemcpyDeviceToHost));

   // First pass reduction, doing only reductions per block
   reduce_max<<<blockSize,gridSize,sizeof(float)*threadsPerBlock>>>(d_logLuminance, numRows*numCols, 
                                                                    d_result_min_max);
   
   // Doing reduction using the output of the previous step to collect the overall maximum
   reduce_max<<<dim3(1,1,1),dim3(powerTwoBlockSize,1,1),
                sizeof(float)*threadsPerBlock>>>(d_result_min_max, numberBlocks, d_result_min_max);

   checkCudaErrors(cudaMemcpy(&max_logLum, d_result_min_max, sizeof(float), cudaMemcpyDeviceToHost));

   checkCudaErrors(cudaFree(d_result_min_max));

   /*2) subtract them to find the range */
   float range_logLum = max_logLum - min_logLum;

   std::cout<<min_logLum<<" "<<max_logLum<<" range: "<<range_logLum<<std::endl;
   /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
   /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

}
