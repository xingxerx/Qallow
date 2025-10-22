#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Parallel prediction on GPU
// Computes sigmoid-based reward for batch of states
int cuda_predict_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);

// Parallel learning on GPU
// Applies TD-like learning updates to batch of states
int cuda_learn_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);

// Parallel emotion regulation on GPU
// Applies emotion/utility regulation to batch of states
int cuda_emotion_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);

#ifdef __cplusplus
}
#endif

