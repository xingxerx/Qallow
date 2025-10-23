#include <cuda_runtime.h>
#include <algorithm>
#include "ccc.hpp"

namespace qallow::ccc {

// ── util: gray decode ──
__device__ __forceinline__ uint32_t d_gray2int(uint32_t g){
  for(uint32_t b=g>>1; b; b>>=1) g ^= b;
  return g;
}
int gray2int(uint32_t g){
  for(uint32_t b=g>>1; b; b>>=1) g ^= b;
  return static_cast<int>(g);
}

// ── H_dyn partials: σ^z coeffs for modes and ctrl ──
__global__ void k_build_cost_coeffs(const float* __restrict__ lambda_m, // [B,M]
                                    const float* __restrict__ c_bits,   // [B,b]
                                    float* __restrict__ hz_mode,        // [B,M]
                                    float* __restrict__ hz_ctrl,        // [B,b]
                                    float alpha, float rho, int M, int b){
  int Bidx = blockIdx.x;
  int i = threadIdx.x;
  if(i < M) hz_mode[Bidx*M + i] = alpha * lambda_m[Bidx*M + i];
  if(i < b) hz_ctrl[Bidx*b + i] = rho   * c_bits[Bidx*b + i];
}
void build_cost_coeffs(const float* lambda_m, const float* c_bits,
                       float* hz_mode, float* hz_ctrl,
                       float alpha, float rho, int B, int M, int b){
  int blk = std::max(M,b);
  k_build_cost_coeffs<<<B, blk>>>(lambda_m, c_bits, hz_mode, hz_ctrl, alpha, rho, M, b);
  cudaDeviceSynchronize();
}

// ── H_temp: Ising chain penalty per bit between t and t+1 ──
// pair_cost[B] accumulates η * Hamming(ctrl_t, ctrl_tp1)
__global__ void k_temporal_chain(const uint8_t* __restrict__ mem_t,  // [B,b]
                                 const uint8_t* __restrict__ mem_tp, // [B,b]
                                 float* __restrict__ out_cost, float eta, int b){
  int Bidx = blockIdx.x;
  int j = threadIdx.x;
  extern __shared__ float ssum_arr[];
  float* ssum = ssum_arr;
  if(j==0) *ssum = 0.f;
  __syncthreads();
  uint8_t bt = (j<b) ? mem_t[Bidx*b + j] : 0;
  uint8_t bp = (j<b) ? mem_tp[Bidx*b + j] : 0;
  float c = (j<b && (bt^bp)) ? 1.f : 0.f;
  atomicAdd(ssum, c);
  __syncthreads();
  if(j==0) out_cost[Bidx] = eta * (*ssum);
}
void build_temporal_chain(const uint8_t* mem_bits_t, const uint8_t* mem_bits_tp1,
                          float* pair_cost, float eta, int B, int b){
  k_temporal_chain<<<B, b, sizeof(float)>>>(mem_bits_t, mem_bits_tp1, pair_cost, eta, b);
  cudaDeviceSynchronize();
}

} // namespace qallow::ccc
