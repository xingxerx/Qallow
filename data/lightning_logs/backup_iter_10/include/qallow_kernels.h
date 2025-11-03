#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int qallow_p12_elasticity_gpu(int N, float scale, float* out_mean);
int qallow_p13_harmonic_gpu(int N, float freq, float* out_energy);

#ifdef __cplusplus
}
#endif
