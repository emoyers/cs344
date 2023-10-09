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

#define MIN(X,Y) (X<Y)?X:Y
#define MAX(X,Y) (X>Y)?X:Y

__global__ void reduce_min_n_max(const float* const d_input, const size_t input_size, const size_t input_offset_max,
                                 float* const d_output, const size_t output_size)
{

   extern __shared__ float sh_data[];
   int array_id = threadIdx.x + blockDim.x * blockIdx.x;
   int thread_id = threadIdx.x;
   int offset_max = blockDim.x;

   // load shared mem from global memory
   if(array_id < input_size)
   {
      // for min
      sh_data[thread_id] = d_input[array_id];

      // for max
      sh_data[offset_max+thread_id] = d_input[array_id+input_offset_max];
   }
   else
   {
      // for min
      sh_data[thread_id] = FLT_MAX;

      // for max
      sh_data[offset_max+thread_id] = FLT_MIN;
   }
   __syncthreads();

   // Do reduction on block level
   for(unsigned int s=blockDim.x/2; s > 0; s>>=1)
   {
      if(thread_id < s)
      {
         // for min
         sh_data[thread_id] = MIN(sh_data[thread_id],sh_data[thread_id+s]);

         // for max
         sh_data[offset_max+thread_id] = MAX(sh_data[offset_max+thread_id],
                                             sh_data[offset_max+thread_id+s]);
      }
      __syncthreads();
   }

   if(thread_id == 0)
   {
      // for min
      d_output[blockIdx.x] = sh_data[0];

      // for max
      d_output[output_size+blockIdx.x] = sh_data[offset_max];
   }

}

__global__ void generate_histogram(const float* const d_input, const size_t input_size,
                                   float min_logLum, float range_logLum,
                                   unsigned int* const d_output, const size_t output_size)
{
   // TODO optimize this histogram generation
   int array_id = threadIdx.x + blockDim.x * blockIdx.x;

   if(array_id < input_size)
   {
      unsigned int bin_id = MIN((unsigned int)(output_size - 1u), 
                                (unsigned int)((d_input[array_id] - min_logLum) / range_logLum * output_size));
      atomicAdd(&(d_output[bin_id]), 1u);
   }

}

__global__ void exclusive_scan_histogram(unsigned int* const d_input_ouput, const size_t input_output_size)
{
   extern __shared__ unsigned int sh_data_[];
   size_t thread_id = threadIdx.x;

   // load shared mem from global memory
   if(thread_id < input_output_size)
   {
      sh_data_[thread_id] = d_input_ouput[thread_id];
   }
   else
   {
      sh_data_[thread_id] = 0u;
   }
   __syncthreads();

   // Doing Hillis/Steele inclusive scan
   size_t power_two_acc = 1u;
   unsigned int temp_read = 0u;
   while(power_two_acc < input_output_size)
   {
      // Read
      if(thread_id>=power_two_acc & thread_id < input_output_size)
      {
         temp_read = sh_data_[thread_id-power_two_acc];
      }
      else
      {
         temp_read = 0u;
      }
      __syncthreads();
         
      sh_data_[thread_id] += temp_read;
      __syncthreads();

      power_two_acc = power_two_acc * 2u;
   }

   // Converting to exclusice scan and copying to device global memory
   if(thread_id == 0)
   {
      d_input_ouput[thread_id] = 0u;
   }
   else if(thread_id < input_output_size)
   {
      d_input_ouput[thread_id] = sh_data_[thread_id-1u];
   }
   else
   {
      // Do nothing
   }
   __syncthreads();


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

   checkCudaErrors(cudaMalloc(&d_result_min_max, 2*powerTwoBlockSize*sizeof(float)));
   
   // First pass reduction, doing only reductions per block. Offset is 0 because they are copying to shared memory
   // from same array same values.
   reduce_min_n_max<<<blockSize,gridSize,2*sizeof(float)*threadsPerBlock>>>(d_logLuminance, numRows*numCols, 0u,
                                                                            d_result_min_max, powerTwoBlockSize);
   // Doing reduction using the output of the previous step to collect the overall minimum and maximum. 
   // Offset is "powerTwoBlockSize" because they are copying to shared memory the output of previous call
   // which has an offset of powerTwoBlockSize for max values.
   reduce_min_n_max<<<dim3(1,1,1),dim3(powerTwoBlockSize,1,1),
                      2*sizeof(float)*powerTwoBlockSize>>>(d_result_min_max, numberBlocks, powerTwoBlockSize, 
                                                           d_result_min_max, powerTwoBlockSize);

   // Getting min and max from device
   checkCudaErrors(cudaMemcpy(&min_logLum, d_result_min_max, sizeof(float), cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaMemcpy(&max_logLum, &d_result_min_max[powerTwoBlockSize], sizeof(float), cudaMemcpyDeviceToHost));

   // Free the device allocated memory
   checkCudaErrors(cudaFree(d_result_min_max));

   /*2) subtract them to find the range */
   float range_logLum = max_logLum - min_logLum;

   /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
   checkCudaErrors(cudaMemset(d_cdf, 0, numBins*sizeof(unsigned int)));

   generate_histogram<<<blockSize,gridSize>>>(d_logLuminance, numRows*numCols, min_logLum, range_logLum, d_cdf, numBins);


   /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   numberBlocks = (int)std::ceil(double(numBins) / (double)threadsPerBlock);
   exclusive_scan_histogram<<<dim3(1,1,1), gridSize, sizeof(unsigned int)*numBins>>>(d_cdf, numBins);
}
