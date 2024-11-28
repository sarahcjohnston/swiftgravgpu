/* This include */
#include "gpu_params.h"

/* Local includes */
#include "cuda_streams.h"
#include "error.h"

/* Cuda inlcudes */
#include <cuda.h>
#include <cuda_runtime.h>

extern void gpu_device_props(struct gpu_info *gpu_info);

gpu_info gpu_init_info(struct swift_params *params) {

  /* Allocate memory for the gpu properties. */
  struct gpu_info *gpu_info =
      (struct gpu_info *)malloc(sizeof(struct gpu_info));

  /* Get all the device properties */
  gpu_device_props(gpu_info);

  /* Get the number of CUDA streams from the parameters */
  gpu_info->nr_streams = parser_get_opt_param_int(params, "GPU:nstreams", 8);

  /* Report what we've found */
  message("GPU device ID: %d", gpu_info->device_id);
  message("Number of SMs: %d", gpu_info->nr_sm);
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
  message("Number of CUDA streams: %d", gpu_info->nr_streams);

  return gpu_info;
}
