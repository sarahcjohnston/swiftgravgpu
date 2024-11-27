#ifndef CUDA_STREAMS_H
#define CUDA_STREAMS_H

#include <cuda_runtime.h>

#define MAX_STREAMS 16

/**
 * @brief A "singleton" structure for holding the CUDA streams.
 *
 * This structure is used to hold the CUDA streams that are created at the
 * beginning of the run.
 *
 * @param streams An array of CUDA streams.
 * @param nstreams The number of CUDA streams created.
 */
struct cuda_streams {
  cudaStream_t streams[MAX_STREAMS]; /*!< The streams themselves. */
  int nstreams;                      /*!< The number of streams created. */
};

/* Declare the global singleton instance */
extern struct cuda_streams *streams;

/* Function prototypes */
int engine_cuda_init_streams(int num_streams);
int destroy_persistent_cuda_streams();
cudaStream_t get_cuda_stream(int index);

#endif  // CUDA_STREAMS_H
