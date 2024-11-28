
/* Local includes */
#include "gpu_params.h"

#include "cuda_streams.h"

/* Cuda inlcudes */
#include <cuda.h>
#include <cuda_runtime.h>

void gpu_init_info(struct gpu_info *gpu_info, struct swift_params *params) {

  /* Allocate memory for the gpu properties. */
  struct gpu_info *gpu_info =
      (struct gpu_info *)malloc(sizeof(struct gpu_info));

  /* Set the device ID */
  cudaGetDevice(&gpu_info->device_id);

  /* Get the device properties */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_info->device_id);

  /* Set the number of streaming multiprocessors */
  gpu_info->num_sm = deviceProp.multiProcessorCount;

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

  /* Get the number of CUDA streams from the parameters */
  gpu_info->nr_cuda_streams = parser_get_opt_param_int(
      params, "GPU:nstreams", space_max_top_level_cells_default);

  /* Report what we've found */
  message("GPU device ID: %d", gpu_info->device_id);
  message("Number of SMs: %d", gpu_info->num_sm);
  message("Max threads per SM: %d", gpu_info->max_threads_per_sm);
  message("Max threads per block: %d", gpu_info->max_threads_per_block);
  message("Max blocks per SM: %d", gpu_info->max_blocks_per_sm);
  message("Max shared memory per SM: %d", gpu_info->max_shared_memory_per_sm);
  message("Max shared memory per block: %d",
          gpu_info->max_shared_memory_per_block);
  message("Max registers per block: %d", gpu_info->max_registers_per_block);
  message("Warp size: %d", gpu_info->warp_size);
  message("Max threads per block dimension: %d",
          gpu_info->max_threads_per_block_dimension);
  message("Max grid size: %d", gpu_info->max_grid_size);
  message("Max threads per block dimension x: %d",
          gpu_info->max_threads_per_block_dimension_x);
  message("Max threads per block dimension y: %d",
          gpu_info->max_threads_per_block_dimension_y);
  message("Max threads per block dimension z: %d",
          gpu_info->max_threads_per_block_dimension_z);
}
