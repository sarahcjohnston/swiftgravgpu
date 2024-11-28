#ifndef GPU_PARAMS_H
#define GPU_PARAMS_H

#include <cuda_runtime.h>

struct gpu_info {
  /*!< The device ID of the GPU. */
  int device_id;

  /*!< The number of streaming multiprocessors on the GPU. */
  int num_sm;

  int max_threads_per_sm;       /*!< The maximum number of threads per SM. */
  int max_threads_per_block;    /*!< The maximum number of threads per block. */
  int max_blocks_per_sm;        /*!< The maximum number of blocks per SM. */
  int max_shared_memory_per_sm; /*!< The maximum amount of shared memory per SM.
                                 */
  int max_shared_memory_per_block; /*!< The maximum amount of shared memory per
                                      block. */
  int max_registers_per_block; /*!< The maximum number of registers per block.
                                */
  int warp_size;               /*!< The warp size of the GPU. */
  int max_threads_per_block_dimension;   /*!< The maximum number of threads per
                                            block dimension. */
  int max_grid_size;                     /*!< The maximum grid size. */
  int max_threads_per_block_dimension_x; /*!< The maximum number of threads per
                                            block dimension x. */
  int max_threads_per_block_dimension_y; /*!< The maximum number of threads per
                                            block dimension y. */
  int max_threads_per_block_dimension_z; /*!< The maximum number of threads per
                                            block dimension z. */

  /*!< The number of CUDA streams. */
  int nr_cuda_streams;
};

void gpu_init_info(struct gpu_info *gpu_info, struct swift_params *params);

#endif  // GPU_PARAMS_H
