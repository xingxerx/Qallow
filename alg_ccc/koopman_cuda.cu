#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cassert>
#include "ccc.hpp"

namespace qallow::ccc {

// ── small-matrix helpers (shared-memory Gauss-Jordan), d <= 64 ──
template<int MAXD>
__device__ void gauss_jordan_solve(float* __restrict__ G,  // dxd (row-major)
                                   float* __restrict__ A,  // dxd (RHS matrix; will be overwritten with K)
                                   int d){
  for(int col=0; col<d; ++col){
    // pivot
    int piv = col;
    float best = fabsf(G[piv*d + col]);
    for(int r=col+1; r<d; ++r){
      float v = fabsf(G[r*d + col]);
      if(v > best){ best=v; piv=r; }
    }
    // swap rows in G and A
    if(piv!=col){
      for(int c=0;c<d;++c){ float t=G[col*d+c]; G[col*d+c]=G[piv*d+c]; G[piv*d+c]=t; }
      for(int c=0;c<d;++c){ float t=A[col*d+c]; A[col*d+c]=A[piv*d+c]; A[piv*d+c]=t; }
    }
    float diag = G[col*d + col] + 1e-8f;
    // normalize pivot row
    float inv = 1.f/diag;
    for(int c=0;c<d;++c){ G[col*d+c]*=inv; }
    for(int c=0;c<d;++c){ A[col*d+c]*=inv; }
    // eliminate others
    for(int r=0;r<d;++r){
      if(r==col) continue;
      float f = G[r*d + col];
      if(f==0.f) continue;
      for(int c=0;c<d;++c){ G[r*d+c] -= f*G[col*d+c]; }
      for(int c=0;c<d;++c){ A[r*d+c] -= f*A[col*d+c]; }
    }
  }
}

// X_t:[T,d], X_tp:[T,d] → G=X^T X (dxd), A=X^T X' (dxd)
template<int MAXD,int MAXT>
__global__ void k_fit_koopman(const float* __restrict__ X_t,
                              const float* __restrict__ X_tp,
                              float* __restrict__ K_out,
                              int T,int d){
  extern __shared__ unsigned char smem_raw[];
  float* G = reinterpret_cast<float*>(smem_raw);           // d*d
  float* A = G + MAXD*MAXD;                                // d*d
  // zero
  for(int i=0;i<d;i++){ for(int j=0;j<d;j++){ G[i*d+j]=0.f; A[i*d+j]=0.f; } }
  // accumulate
  for(int t=0;t<T;t++){
    const float* xt  = X_t  + t*d;
    const float* xtp = X_tp + t*d;
    for(int i=0;i<d;i++){
      float xi = xt[i];
      for(int j=0;j<d;j++){
        G[i*d+j] += xi * xt[j];
        A[i*d+j] += xi * xtp[j];
      }
    }
  }
  // solve G * K = A (column-wise in A)
  gauss_jordan_solve<MAXD>(G, A, d);
  // write K=A
  float* K = K_out;
  for(int i=0;i<d;i++) for(int j=0;j<d;j++) K[i*d+j] = A[i*d+j];
}

// Batched launcher
void fit_koopman_batched(const float* X_t, const float* X_tp, float* K_out, int B,int T,int d){
  // layout: batch-major contiguous blocks of [T,d]
  const int MAXD = 64, MAXT = 1024;
  assert(d<=MAXD && T<=MAXT);
  size_t smem = (size_t)(MAXD*MAXD*2)*sizeof(float);
  for(int b=0;b<B;b++){
    const float* xt  = X_t  + (size_t)b*T*d;
    const float* xtp = X_tp + (size_t)b*T*d;
    float*       Kb  = K_out+ (size_t)b*d*d;
    k_fit_koopman<64,1024><<<1,1,smem>>>(xt, xtp, Kb, T, d);
  }
  cudaDeviceSynchronize();
}

// Lyapunov proxy: mean log Frobenius norm, replicated across first M modes
__global__ void k_lyap_frob(const float* __restrict__ J_t, float* __restrict__ Lmb, int T,int d,int M){
  // J_t layout: [T,d,d] for one batch
  float acc = 0.f;
  for(int t=0;t<T;t++){
    const float* J = J_t + t*d*d;
    float frob2 = 0.f;
    for(int i=0;i<d*d;i++){ float v=J[i]; frob2 += v*v; }
    float frob = sqrtf(frob2) + 1e-8f;
    acc += logf(frob);
  }
  float mean_log = acc / T;
  for(int m=0;m<M;m++) Lmb[m] = mean_log;
}
void lyap_jacobian_norms(const float* J_t, float* lmbda, int B,int T,int d,int M){
  for(int b=0;b<B;b++){
    const float* Jb = J_t + (size_t)b*T*d*d;
    float* Lb = lmbda + (size_t)b*M;
    k_lyap_frob<<<1,1>>>(Jb, Lb, T, d, M);
  }
  cudaDeviceSynchronize();
}

// Ethics: sigmoid(W·feat + b)
__global__ void k_ethics_sigmoid(const float* __restrict__ feats, // [T,F]
                                 const float* __restrict__ wb,    // [F+1]
                                 float* __restrict__ out,         // [T]
                                 int T,int F){
  for(int t=0;t<T;t++){
    const float* ft = feats + t*F;
    float s = wb[F]; // bias
    for(int f=0; f<F; ++f) s += wb[f] * ft[f];
    out[t] = 1.f / (1.f + expf(-s));
  }
}
void ethics_score_forward(const float* feats, const float* wb, float* E_out, int B,int T,int F){
  for(int b=0;b<B;b++){
    const float* fb = feats + (size_t)b*T*F;
    const float* w  = wb; // shared across batch
    float* Eb = E_out + (size_t)b*T;
    k_ethics_sigmoid<<<1,1>>>(fb, w, Eb, T, F);
  }
  cudaDeviceSynchronize();
}

// Gray decode per row of bits[ B x b ], LSB = bits[:,0]
__global__ void k_gray2int(const uint8_t* __restrict__ bits, int* __restrict__ val, int b){
  int Bidx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t g = 0;
  for(int j=b-1;j>=0;j--){
    uint8_t bj = bits[Bidx*b + j] & 1u;
    g = (g<<1) | bj;
  }
  uint32_t x = g;
  for(uint32_t t=g>>1; t; t>>=1) x ^= t;
  val[Bidx] = static_cast<int>(x);
}
void gray2int_batch(const uint8_t* bits, int* val, int B,int b){
  int tpB = 128;
  int nBl = (B + tpB - 1)/tpB;
  k_gray2int<<<nBl, tpB>>>(bits, val, b);
  cudaDeviceSynchronize();
}

// Reward gradient toy: slope around center of range [0, vmax]
__global__ void k_reward_grad(const int* __restrict__ val, float* __restrict__ cj, int b, int vmax){
  int Bidx = blockIdx.x;
  int j    = threadIdx.x;
  int v = val[Bidx];
  float centered = ( (float)v - 0.5f*(float)vmax ) / (0.5f*(float)vmax + 1e-6f);
  if(j<b) cj[Bidx*b + j] = centered;
}
void reward_grad_bits(const int* val, float* cj, int B,int b, int vmax){
  k_reward_grad<<<B, b>>>(val, cj, b, vmax);
  cudaDeviceSynchronize();
}

} // namespace qallow::ccc
