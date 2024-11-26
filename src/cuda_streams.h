
/* Cuda includes */
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_STREAMS 16

/**
 * @brief A "singlegton" structure for holding the CUDA streams.
 *
 * This structure is used to hold the CUDA streams that are created at the
 * beginning of the run.
 *
 * @param streams An array of CUDA streams.
 * @param streams_created A flag indicating if the streams have been created.
 */
struct cuda_streams {

  /*! The streams themselves. */
  cudaStream_t streams[MAX_STREAMS];

  /*! The number of streams created. */
  int nstreams;
};

extern struct cuda_streams *streams;

/**
 * @brief Function to create the CUDA streams.
 *
 * This should be called once at the beginning of time to create the CUDA
 * streams we'll interleave operations on.
 *
 * These must be destroyed with destroy_persistent_cuda_streams() when done.
 *
 * Note that the streams are stored in a "singleton" structure, so this
 * function can be called multiple times, but the streams will only be created
 * once.
 *
 * @param num_streams The number of CUDA streams to create.
 * @return The number of streams created.
 */
int engine_cuda_init_streams(int num_streams) {
  /* Check if the streams have already been created */
  if (streams->nstreams) {
    /* If the streams are already created, return the number of streams */
    return streams->nstreams;
  }

  int i;
  /* Allocate and initialize an array of CUDA streams */
  for (i = 0; i < num_streams && i < MAX_STREAMS; i++) {
    cudaError_t err =
        cudaStreamCreateWithFlags(&streams->streams[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      /* If unable to create a stream, free previously created streams and
       * return an error code */
      for (int j = 0; j < i; j++) {
        cudaStreamDestroy(streams->streams[j]);
      }
      return 0;
    }
  }

  /* Set the flag indicating the streams have been successfully created */
  streams->nstreams = i;

  /* Return the number of streams created */
  return streams->nstreams;
}

/**
 * @brief Function to destroy the CUDA streams.
 *
 * This function is used to destroy the CUDA streams that were created at the
 * beginning of the run.
 */
int destroy_persistent_cuda_streams() {
  /* Check if the streams have been created */
  if (!streams->nstreams) {
    /* If the streams have not been created, return an error code */
    return 0;
  }

  /* Destroy the CUDA streams */
  for (int i = 0; i < streams->nstreams; i++) {
    cudaStreamDestroy(streams->streams[i]);
  }

  /* Reset the flag indicating the streams have been created */
  streams->nstreams = 0;

  /* Return success */
  return 0;
}
