#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <stdio.h>
extern "C" {
#include "qallow.h"
#include "pocket.h"
}

// CUDA error checking macro
#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if(e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return -1; \
    } \
} while(0)

// Global state
static double *d_orb = nullptr, *d_riv = nullptr, *d_myc = nullptr;
static double *d_orb_mean = nullptr, *d_riv_mean = nullptr, *d_myc_mean = nullptr;
static int G_P = 0, G_N = 0;

/* simple linear congruential generator */
static __device__ __inline__ double lcg(uint32_t* s){
    *s = (*s * 1664525u + 1013904223u);
    return ((*s >> 8) & 0xFFFFFF) / double(0xFFFFFF);
}

/* initialization kernel */
__global__ void k_init(double* orb, double* riv, double* myc, int P, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if(i < N && p < P){
        int idx = p * N + i;
        uint32_t seed = (uint32_t)(clock64() + idx);
        orb[idx] = 0.9 + 0.05 * lcg(&seed);
        riv[idx] = 0.995 + 0.005 * lcg(&seed);
        myc[idx] = 0.9995 + 0.0005 * lcg(&seed);
    }
}

/* update kernel - simulates pocket evolution */
__global__ void k_update(double* O, double* R, double* M, int P, int N, double jitter, int tick){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if(i < N && p < P){
        int idx = p * N + i;
        uint32_t seed = (uint32_t)(clock64() + idx + tick);
        double j = (lcg(&seed) - 0.5) * jitter;
        
        double o = O[idx] + 0.0001 + j*0.02;
        double r = R[idx] - 0.0002 + j*0.1;
        double m = M[idx] - 0.00001 + j*0.05;
        
        // clamp
        o = fmin(fmax(o, 0.90), 0.95);
        r = fmin(fmax(r, 0.995), 1.000);
        m = fmin(fmax(m, 0.9995), 1.0000);
        
        O[idx] = o;
        R[idx] = r;
        M[idx] = m;
    }
}

// reduction: mean over pockets for each node
__global__ void k_mean_over_pockets(const double* __restrict__ X, double* __restrict__ OUT, int P, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    double acc = 0.0;
    for(int p = 0; p < P; ++p) acc += X[p * N + i];
    OUT[i] = acc / (double)P;
}

extern "C" int pocket_spawn_and_run(const pocket_cfg_t* cfg){
  if(!cfg || cfg->pockets<=0 || cfg->nodes<=0 || cfg->steps<=0) return -1;
  G_P = cfg->pockets; G_N = cfg->nodes;

  size_t PN = (size_t)G_P * (size_t)G_N;
  CUDA_OK(cudaMalloc(&d_orb, PN*sizeof(double)));
  CUDA_OK(cudaMalloc(&d_riv, PN*sizeof(double)));
  CUDA_OK(cudaMalloc(&d_myc, PN*sizeof(double)));
  CUDA_OK(cudaMalloc(&d_orb_mean, G_N*sizeof(double)));
  CUDA_OK(cudaMalloc(&d_riv_mean, G_N*sizeof(double)));
  CUDA_OK(cudaMalloc(&d_myc_mean, G_N*sizeof(double)));

  dim3 block(128,1);              // nodes x pockets split
  dim3 grid((G_N+block.x-1)/block.x, G_P); // one warp-row per pocket

  k_init<<<grid, block>>>(d_orb, d_riv, d_myc, G_P, G_N);
  CUDA_OK(cudaGetLastError());

  // stream per pocket (optional; kernel already 2D). Example multi-stream loop:
  // Here we keep single kernel per tick for simplicity and good occupancy.
  for(int t=0;t<cfg->steps;++t){
    k_update<<<grid, block>>>(d_orb, d_riv, d_myc, G_P, G_N, cfg->jitter, t);
    CUDA_OK(cudaGetLastError());
  }
  CUDA_OK(cudaDeviceSynchronize());

  // means per node
  dim3 block1(256);
  dim3 grid1((G_N+block1.x-1)/block1.x);
  k_mean_over_pockets<<<grid1, block1>>>(d_orb, d_orb_mean, G_P, G_N);
  k_mean_over_pockets<<<grid1, block1>>>(d_riv, d_riv_mean, G_P, G_N);
  k_mean_over_pockets<<<grid1, block1>>>(d_myc, d_myc_mean, G_P, G_N);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  return 0;
}

extern "C" int pocket_merge_to_host(double* orbital, double* river, double* mycelial){
  if(!d_orb_mean||!d_riv_mean||!d_myc_mean) return -1;
  CUDA_OK(cudaMemcpy(orbital,  d_orb_mean, G_N*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(river,    d_riv_mean, G_N*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(mycelial, d_myc_mean, G_N*sizeof(double), cudaMemcpyDeviceToHost));
  return 0;
}

extern "C" int pocket_release(){
  if(d_orb)       cudaFree(d_orb), d_orb=nullptr;
  if(d_riv)       cudaFree(d_riv), d_riv=nullptr;
  if(d_myc)       cudaFree(d_myc), d_myc=nullptr;
  if(d_orb_mean)  cudaFree(d_orb_mean), d_orb_mean=nullptr;
  if(d_riv_mean)  cudaFree(d_riv_mean), d_riv_mean=nullptr;
  if(d_myc_mean)  cudaFree(d_myc_mean), d_myc_mean=nullptr;
  G_P=G_N=0;
  return 0;
}
