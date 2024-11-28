
#include "gpu_params.h"


void gpu_device_props(struct gpu_info *gpu_info) {

  /* Set the device ID */
  cudaGetDevice(&gpu_info->device_id);

  /* Get the device properties */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_info->device_id);

  /* Set the number of streaming multiprocessors */
  gpu_info->nr_sm = deviceProp.multiProcessorCount;

  /* Set the maximum number of threads per SM */
  gpu_info->max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;

  /* Set the maximum number of threads per block */
  gpu_info->max_threads_per_block = deviceProp.maxThreadsPerBlock;

  /* Set the maximum number of blocks per SM */
  gpu_info->max_blocks_per_sm = deviceProp.maxBlocksPerMultiProcessor;

  /* Set the maximum amount of shared memory per SM */
  gpu_info->max_shared_memory_per_sm = deviceProp.sharedMemPerMultiprocessor;

  /* Set the maximum amount of shared memory per block */
  gpu_info->max_shared_memory_per_block = deviceProp.sharedMemPerBlock;

  /* Set the maximum number of registers per block */
  gpu_info->max_registers_per_block = deviceProp.regsPerBlock;

  /* Set the warp size */
  gpu_info->warp_size = deviceProp.warpSize;

  /* Set the maximum number of threads per block dimension */
  gpu_info->max_threads_per_block_dimension = deviceProp.maxThreadsDim[0];

  /* Set the maximum grid size */
  gpu_info->max_grid_size = deviceProp.maxGridSize[0];

  /* Set the maximum number of threads per block dimension x */
  gpu_info->max_threads_per_block_dimension_x = deviceProp.maxThreadsDim[0];

  /* Set the maximum number of threads per block dimension y */
  gpu_info->max_threads_per_block_dimension_y = deviceProp.maxThreadsDim[1];

  /* Set the maximum number of threads per block dimension z */
  gpu_info->max_threads_per_block_dimension_z = deviceProp.maxThreadsDim[2];
}


