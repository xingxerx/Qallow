#pragma once

__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, o);
    }
    return v;
}

template <int BS>
__device__ float block_sum(float v) {
    __shared__ float s[BS / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    v = (threadIdx.x < BS / 32) ? s[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}
