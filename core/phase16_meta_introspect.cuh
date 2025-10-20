#pragma once

#ifdef __CUDACC__
extern "C" __global__
void introspect_kernel(const float* __restrict__ duration,
                       const float* __restrict__ coherence,
                       const float* __restrict__ ethics,
                       float* __restrict__ scores,
                       int count);
#endif
