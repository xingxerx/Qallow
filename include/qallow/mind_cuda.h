#pragma once

#ifdef __cplusplus
extern "C" {
#endif



int cuda_predict_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);



int cuda_learn_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);



int cuda_emotion_batch(double *h_energy, double *h_risk, double *h_reward, int batch_size);

#ifdef __cplusplus
}
#endif

