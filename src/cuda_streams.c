/* This includes */
#include "cuda_streams.h"

#include <stdio.h>
#include <stdlib.h>

/* Define the global singleton instance */
struct cuda_streams *streams = NULL;

/**
 * @brief Function to create the CUDA streams.
 *
 * This should be called once at the beginning to create the CUDA
 * streams we'll interleave operations on.
 *
 * These must be destroyed with destroy_persistent_cuda_streams() when done.
 *
 * @param num_streams The number of CUDA streams to create.
 * @return The number of streams created.
 */
int engine_cuda_init_streams(int num_streams) {
  if (streams == NULL) {
    // Allocate memory for the singleton structure
    streams = (struct cuda_streams *)malloc(sizeof(struct cuda_streams));
    if (streams == NULL) {
      fprintf(stderr,
              "Failed to allocate memory for CUDA streams singleton.\n");
      return 0;
    }
    streams->nstreams = 0;
  }

  /* Check if the streams have already been created */
  if (streams->nstreams) {
    /* If the streams are already created, return the number of streams */
    return streams->nstreams;
  }

  /* Allocate and initialize an array of CUDA streams */
  int i;
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

  /* Set the number of streams created */
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
  if (streams == NULL || streams->nstreams == 0) {
    /* If the streams have not been created, return an error code */
    return 0;
  }

  /* Destroy the CUDA streams */
  for (int i = 0; i < streams->nstreams; i++) {
    cudaStreamDestroy(streams->streams[i]);
  }

  /* Reset the number of streams created */
  streams->nstreams = 0;

  /* Free the singleton structure */
  free(streams);
  streams = NULL;

  /* Return success */
  return 0;
}

/**
 * @brief Function to get a CUDA stream.
 *
 * @param index The index of the CUDA stream to get.
 * @return The CUDA stream at the given index.
 */
cudaStream_t get_cuda_stream(int index) {
  if (streams != NULL && index < streams->nstreams && index >= 0) {
    return streams->streams[index];
  }
  return NULL;
}
