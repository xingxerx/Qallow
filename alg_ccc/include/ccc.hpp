#pragma once
#include <cstdint>

namespace qallow::ccc {

// Gray → int (host)
int gray2int(uint32_t g);

// Params
struct CCCParams {
  int M_modes{8};
  int b_ctrl{6};
  int H_horizon{4};
  float alpha{1.f}, beta{1.f}, rho{0.1f}, gamma{5.f}, eta{1.f}, kappa{0.1f}, xi{0.1f};
  float ethics_tau{0.94f};
};

// ── GPU entrypoints ──
// Koopman: K = (X^T X)^{-1} (X^T X')
void fit_koopman_batched(const float* X_t, const float* X_tp, float* K_out, int B, int T, int d);
// Approx. Lyapunov exponents (top-M proxy from log‖J‖_F mean)
void lyap_jacobian_norms(const float* J_t, float* lmbda, int B, int T, int d, int M);
// Ethics score: sigmoid(W·feat + b) per time
void ethics_score_forward(const float* feats, const float* wb, float* E_out, int B, int T, int F);

// Bits helpers
void gray2int_batch(const uint8_t* bits, int* val, int B, int b);

// Reward grad per bit (toy linear slope around center)
void reward_grad_bits(const int* val, float* cj, int B, int b, int vmax);

// Hamiltonian term builders
void build_cost_coeffs(const float* lambda_m, const float* c_bits,
                       float* hz_mode, float* hz_ctrl,
                       float alpha, float rho, int B, int M, int b);

void build_temporal_chain(const uint8_t* mem_bits_t, const uint8_t* mem_bits_tp1,
                          float* pair_cost, float eta, int B, int b);

} // namespace qallow::ccc
