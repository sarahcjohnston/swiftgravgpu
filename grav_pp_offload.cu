#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "externalfunctions.cu"
#include "multipole_struct.h"

/* Local Cuda includes */ 
#include "src/cuda_streams.h"
#include "src/gpu_params.h"

extern "C" void gpu_device_props(struct gpu_info *gpu_info) {

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




//PP ALL INTERACTIONS
__global__ void pair_grav_pp(int periodic, const float *CoM_i, const float *CoM_j, float rmax_i, float rmax_j, double min_trunc, int *active_i, int *mpole_i, int *active_j, int *mpole_j, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_i_arr, float *mass_j_arr, const float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *a_x_j, float *a_y_j, float *a_z_j, float *pot_i, float *pot_j, int gcount_i, int gcount_padded_i, int gcount_j, int gcount_padded_j, int ci_active, int cj_active, const int symmetric, const int allow_mpole, const struct multipole *restrict multi_i, const struct multipole *restrict multi_j, float *epsilon, const int allow_multipole_j, const int allow_multipole_i) {

	int max_r_decision = 0;

      /* Can we use the Newtonian version or do we need the truncated one ? */

    /* Not periodic -> Can always use Newtonian potential */
    /* Let's updated the active cell(s) only */

      /* First the P2P */
      grav_pp_full(active_i, mpole_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);

      /* Then the M2P */
      grav_pm_full(active_i, mpole_i, gcount_padded_i, CoM_j, multi_j, periodic, dim_0, dim_1, dim_2, x_i, y_i, z_i, gcount_i, a_x_i, a_y_i, a_z_i, *h_i, pot_i, allow_multipole_j, allow_multipole_i, ci_active, 0, symmetric, max_r_decision);

      /* First the P2P */
      grav_pp_full(active_j, mpole_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

      /* Then the M2P */
      grav_pm_full(active_j, mpole_j, gcount_padded_j, CoM_i, multi_i, periodic, dim_0, dim_1, dim_2, x_j, y_j, z_j, gcount_j, a_x_j, a_y_j, a_z_j, *h_j, pot_j, allow_multipole_i, allow_multipole_j, 0, cj_active, symmetric, max_r_decision);

    /* Periodic BC */

    /* Get the relative distance between the CoMs */
    double d[3] = {CoM_j[0] - CoM_i[0], CoM_j[1] - CoM_i[1],
                    CoM_j[2] - CoM_i[2]};

    /* Correct for periodic BCs */
    d[0] = nearestf1(d[0], dim_0);
    d[1] = nearestf1(d[1], dim_1);
    d[2] = nearestf1(d[2], dim_2);

    const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

    /* Get the maximal distance between any two particles */
    const double max_r = sqrt(r2) + rmax_i + rmax_j;

    if (max_r > min_trunc) {
	max_r_decision = 0;}
    else {
	max_r_decision = 1;}

    /* Do we need to use the truncated interactions ? */
    

      /* Periodic but far-away cells must use the truncated potential */

      /* Let's updated the active cell(s) only */

        /* First the (truncated) P2P */
        grav_pp_truncated(active_i, mpole_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);
	

        /* Then the M2P */
        grav_pm_truncated(active_i, mpole_i, gcount_padded_i, CoM_j, multi_j, periodic, dim_0, dim_1, dim_2, r_s_inv, x_i, y_i, z_i, gcount_i, a_x_i, a_y_i, a_z_i, *h_i, pot_i, allow_multipole_j, allow_multipole_i, ci_active, 0, symmetric, max_r_decision);

        /* First the (truncated) P2P */
	grav_pp_truncated(active_j, mpole_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

        /* Then the M2P */
	grav_pm_truncated(active_j, mpole_j, gcount_padded_j, CoM_i, multi_i, periodic, dim_0, dim_1, dim_2, r_s_inv, x_i, y_i, z_i, gcount_j, a_x_j, a_y_j, a_z_j, *h_j, pot_j, allow_multipole_j, allow_multipole_i, 0, cj_active, symmetric, max_r_decision);

      /* Periodic but close-by cells can use the full Newtonian potential */

      /* Let's updated the active cell(s) only */

        /* First the (Newtonian) P2P */
        grav_pp_full(active_i, mpole_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);

        /* Then the M2P */
        grav_pm_full(active_i, mpole_i, gcount_padded_i, CoM_j, multi_j, periodic, dim_0, dim_1, dim_2, x_i, y_i, z_i, gcount_i, a_x_i, a_y_i, a_z_i, *h_i, pot_i, allow_multipole_j, allow_multipole_i, ci_active, 0, symmetric, max_r_decision);

        /* First the (Newtonian) P2P */
        grav_pp_full(active_j, mpole_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

        /* Then the M2P */
        grav_pm_full(active_j, mpole_j, gcount_padded_j, CoM_i, multi_i, periodic, dim_0, dim_1, dim_2, x_j, y_j, z_j, gcount_j, a_x_j, a_y_j, a_z_j, *h_j, pot_j, allow_multipole_i, allow_multipole_j, 0, cj_active, symmetric, max_r_decision);

}


//do not touch these variables you dumbass you need them to be pointers girly
extern "C" void pp_offload(int periodic, const float *CoM_i, const float *CoM_j, float rmax_i, float rmax_j, double min_trunc, int* active_i, int* mpole_i, int* active_j, int* mpole_j, float *dim, const float *x_i, const float *x_j_arr, const float *y_i, const float *y_j_arr, const float *z_i, const float *z_j_arr, float *pot_i, float *pot_j, float *a_x_i, float *a_y_i, float *a_z_i, float *a_x_j, float *a_y_j, float *a_z_j, float *mass_i_arr, float *mass_j_arr, const float *r_s_inv, float *h_i, float *h_j_arr, const int *gcount_i, const int *gcount_padded_i, const int *gcount_j, const int *gcount_padded_j, int ci_active, int cj_active, const int symmetric, const int allow_mpole, const struct multipole *restrict multi_i, const struct multipole *restrict multi_j, float *epsilon, const int *allow_multipole_j, const int *allow_multipole_i, struct gpu_info *gpu_info){

	
	float a_x_i_new[*gcount_i];
	float a_y_i_new[*gcount_i];
	float a_z_i_new[*gcount_i];
	float pot_i_new[*gcount_i];

	float a_x_j_new[*gcount_j];
	float a_y_j_new[*gcount_j];
	float a_z_j_new[*gcount_j];
	float pot_j_new[*gcount_j];

	//create device pointers
	float *d_h_i;
	float *d_h_j;
	float *d_mass_i;
	float *d_mass_j;
	float *d_x_i;
	float *d_x_j;
	float *d_y_i;
	float *d_y_j;
	float *d_z_i;
	float *d_z_j;
	float *d_a_x_i;
	float *d_a_y_i;
	float *d_a_z_i;
	float *d_a_x_j;
	float *d_a_y_j;
	float *d_a_z_j;
	float *d_pot_i;
	float *d_pot_j;
	int *d_active_i;
	int *d_mpole_i;
	int *d_active_j;
	int *d_mpole_j;
	float *d_CoM_i;
	float *d_CoM_j;

	/* Get a stream to use (for testing these run 0-3 inclusive), we'll 
	 * randomly select one for now. */
	int rand_stream = rand() % gpu_info->nr_streams;
	cudaStream_t stream = streams->streams[rand_stream];

	//cudaDeviceSynchronize();

        //cudaMalloc(&h_multi_j, 13*sizeof(float));
        //cudaMemcpyAsync(h_multi_j, multi_j, 13*sizeof(float), cudaMemcpyHostToDevice);
	multipole* d_multi_j;
     	cudaMallocAsync(&d_multi_j, sizeof(multipole), stream);
     	cudaMemcpyAsync(d_multi_j, multi_j, sizeof(multipole), cudaMemcpyHostToDevice, stream);
	multipole* d_multi_i;
     	cudaMallocAsync(&d_multi_i, sizeof(multipole), stream);
     	cudaMemcpyAsync(d_multi_i, multi_i, sizeof(multipole), cudaMemcpyHostToDevice, stream);

	//allocate memory on device
	cudaMallocAsync(&d_h_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_h_j, *gcount_padded_j * sizeof(float), stream);
	cudaMallocAsync(&d_mass_i, *gcount_padded_i * sizeof(float), stream);
	cudaMallocAsync(&d_mass_j, *gcount_padded_j * sizeof(float), stream);
	cudaMallocAsync(&d_x_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_x_j, *gcount_padded_j * sizeof(float), stream);
	cudaMallocAsync(&d_y_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_y_j, *gcount_padded_j * sizeof(float), stream);
	cudaMallocAsync(&d_z_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_z_j, *gcount_padded_j * sizeof(float), stream);
	cudaMallocAsync(&d_a_x_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_a_y_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_a_z_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_a_x_j, *gcount_j * sizeof(float), stream);
	cudaMallocAsync(&d_a_y_j, *gcount_j * sizeof(float), stream);
	cudaMallocAsync(&d_a_z_j, *gcount_j * sizeof(float), stream);
	cudaMallocAsync(&d_pot_i, *gcount_i * sizeof(float), stream);
	cudaMallocAsync(&d_pot_j, *gcount_j * sizeof(float), stream);
	cudaMallocAsync(&d_active_i, *gcount_i * sizeof(int), stream);
	cudaMallocAsync(&d_mpole_i, *gcount_i * sizeof(int), stream);
	cudaMallocAsync(&d_active_j, *gcount_j * sizeof(int), stream);
	cudaMallocAsync(&d_mpole_j, *gcount_j * sizeof(int), stream);
	cudaMallocAsync(&d_CoM_i, 3 * sizeof(float), stream);
	cudaMallocAsync(&d_CoM_j, 3 * sizeof(float), stream);

	//copy data to device
	cudaMemcpyAsync(d_h_i, h_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_h_j, h_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_mass_i, mass_i_arr, *gcount_padded_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_mass_j, mass_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_x_i, x_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_x_j, x_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_y_i, y_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_y_j, y_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_z_i, z_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_z_j, z_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_x_i, a_x_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_y_i, a_y_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_z_i, a_z_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_x_j, a_x_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_y_j, a_y_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_a_z_j, a_z_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_pot_i, pot_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_pot_j, pot_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_active_i, active_i, *gcount_i * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_mpole_i, mpole_i, *gcount_i * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_active_j, active_j, *gcount_j * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_mpole_j, mpole_j, *gcount_j * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_CoM_i, CoM_i, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_CoM_j, CoM_j, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
	
	//printf("%.16f %.16f\n", x_i[0], y_i[0]);

	cudaError_t err = cudaGetLastError();
    	if (err != cudaSuccess) 
	printf("Error1: %s\n", cudaGetErrorString(err));

        //cudaDeviceSynchronize();

	//call kernel function
	pair_grav_pp<<<gpu_info->sms_multiple * gpu_info->nr_sm, 256, 0, stream>>>(periodic, d_CoM_i, d_CoM_j, rmax_i, rmax_j, min_trunc, d_active_i, d_mpole_i, d_active_j, d_mpole_j, dim[0], dim[1], dim[2], d_h_i, d_h_j, d_mass_i, d_mass_j, *r_s_inv, d_x_i, d_x_j, d_y_i, d_y_j, d_z_i, d_z_j, d_a_x_i, d_a_y_i, d_a_z_i, d_a_x_j, d_a_y_j, d_a_z_j, d_pot_i, d_pot_j, *gcount_i, *gcount_padded_i, *gcount_j, *gcount_padded_j, ci_active, cj_active, symmetric, allow_mpole, d_multi_i, d_multi_j, epsilon, *allow_multipole_j, *allow_multipole_i);

        //cudaDeviceSynchronize();

	cudaError_t err2 = cudaGetLastError();
    	if (err2 != cudaSuccess)
	printf("Error2: %s\n", cudaGetErrorString(err2));

	//copy data from device
	cudaMemcpyAsync(&a_x_i_new, d_a_x_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&a_y_i_new, d_a_y_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&a_z_i_new, d_a_z_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&pot_i_new, d_pot_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost, stream);

	cudaMemcpyAsync(&a_x_j_new, d_a_x_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&a_y_j_new, d_a_y_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&a_z_j_new, d_a_z_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&pot_j_new, d_pot_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost, stream);

        //printf("%.16f %.16f %.16f %.16f\n", a_x_i_new[0], a_y_i_new[0], a_z_i_new[0], pot_i_new[0]);

	cudaStreamSynchronize(stream);

	cudaError_t err3 = cudaGetLastError();
    	if (err3 != cudaSuccess)
	printf("Error3: %s\n", cudaGetErrorString(err3));

	for (int i = 0; i < *gcount_i; i++){
		a_x_i[i] += a_x_i_new[i];
		a_y_i[i] += a_y_i_new[i];
		a_z_i[i] += a_z_i_new[i];
		pot_i[i] += pot_i_new[i];}
		
	for (int i = 0; i < *gcount_j; i++){
		a_x_j[i] += a_x_j_new[i];
		a_y_j[i] += a_y_j_new[i];
		a_z_j[i] += a_z_j_new[i];
		pot_j[i] += pot_j_new[i];}

	
	/*printf("gpu gcount_i: %i ", *gcount_i);
  	for (int i = 0; i < *gcount_i; i++){
  		printf("%.16f ", a_x_i[i]);}
  	printf("\n");*/

	//printf("%f %f %f \n", a_x_i[0], a_y_i[0], a_z_i[0]);

	//printf("RESULT2: %f %f %f %f ", a_x_new[0], a_y_new[0], a_z_new[0], pot_new[0]);

	//free memory
	cudaFreeAsync(d_h_i, stream);
	cudaFreeAsync(d_h_j,stream);
	cudaFreeAsync(d_mass_i,stream);
	cudaFreeAsync(d_mass_j,stream);
	cudaFreeAsync(d_x_i,stream);
	cudaFreeAsync(d_x_j,stream);
	cudaFreeAsync(d_y_i,stream);
	cudaFreeAsync(d_y_j,stream);
	cudaFreeAsync(d_z_i,stream);
	cudaFreeAsync(d_z_j,stream);
	cudaFreeAsync(d_a_x_i,stream);
	cudaFreeAsync(d_a_y_i,stream);
	cudaFreeAsync(d_a_z_i,stream);
	cudaFreeAsync(d_a_x_j,stream);
	cudaFreeAsync(d_a_y_j,stream);
	cudaFreeAsync(d_a_z_j,stream);
	cudaFreeAsync(d_pot_i,stream);
	cudaFreeAsync(d_pot_j,stream);
	cudaFreeAsync(d_active_i,stream);
	cudaFreeAsync(d_mpole_i,stream);
	cudaFreeAsync(d_active_j,stream);
	cudaFreeAsync(d_mpole_j,stream);
	cudaFreeAsync(d_CoM_i,stream);
	cudaFreeAsync(d_CoM_j,stream);
	cudaFreeAsync(d_multi_j,stream);
	cudaFreeAsync(d_multi_i,stream);

	cudaError_t err4 = cudaGetLastError();
    	if (err4 != cudaSuccess)
	printf("Error4: %s\n", cudaGetErrorString(err4));
	
}
