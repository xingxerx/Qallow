#include "ccc.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace qallow::ccc;

// Helper: allocate GPU memory
template<typename T>
T* gpu_alloc(size_t n) {
  T* ptr;
  cudaMalloc(&ptr, n * sizeof(T));
  return ptr;
}

// Helper: copy to GPU
template<typename T>
void to_gpu(T* dst, const T* src, size_t n) {
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
}

// Helper: copy from GPU
template<typename T>
void from_gpu(T* dst, const T* src, size_t n) {
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
}

// Test 1: Gray code decoding
void test_gray_codes() {
  std::cout << "Testing Gray code decoder..." << std::endl;
  
  // Host tests
  assert(gray2int(0b000) == 0);
  assert(gray2int(0b001) == 1);
  assert(gray2int(0b011) == 2);
  assert(gray2int(0b010) == 3);
  assert(gray2int(0b110) == 4);
  assert(gray2int(0b111) == 5);
  assert(gray2int(0b101) == 6);
  assert(gray2int(0b100) == 7);
  
  std::cout << "  ✓ Host gray2int tests passed" << std::endl;
  
  // GPU batch test
  int B = 4, b = 3;
  std::vector<uint8_t> bits_h(B * b);
  bits_h = {0,0,0, 1,0,0, 1,1,0, 0,1,0};  // Gray codes for 0,1,2,3
  
  uint8_t* bits_d = gpu_alloc<uint8_t>(B * b);
  int* val_d = gpu_alloc<int>(B);
  
  to_gpu(bits_d, bits_h.data(), B * b);
  gray2int_batch(bits_d, val_d, B, b);
  
  std::vector<int> val_h(B);
  from_gpu(val_h.data(), val_d, B);
  
  assert(val_h[0] == 0);
  assert(val_h[1] == 1);
  assert(val_h[2] == 2);
  assert(val_h[3] == 3);
  
  cudaFree(bits_d);
  cudaFree(val_d);
  
  std::cout << "  ✓ GPU gray2int_batch tests passed" << std::endl;
}

// Test 2: Cost coefficient builder
void test_cost_coeffs() {
  std::cout << "Testing cost coefficient builder..." << std::endl;
  
  int B = 2, M = 4, b = 3;
  float alpha = 1.5f, rho = 0.2f;
  
  std::vector<float> lambda_m_h(B * M, 0.5f);
  std::vector<float> c_bits_h(B * b, 0.3f);
  
  float* lambda_m_d = gpu_alloc<float>(B * M);
  float* c_bits_d = gpu_alloc<float>(B * b);
  float* hz_mode_d = gpu_alloc<float>(B * M);
  float* hz_ctrl_d = gpu_alloc<float>(B * b);
  
  to_gpu(lambda_m_d, lambda_m_h.data(), B * M);
  to_gpu(c_bits_d, c_bits_h.data(), B * b);
  
  build_cost_coeffs(lambda_m_d, c_bits_d, hz_mode_d, hz_ctrl_d, alpha, rho, B, M, b);
  
  std::vector<float> hz_mode_h(B * M), hz_ctrl_h(B * b);
  from_gpu(hz_mode_h.data(), hz_mode_d, B * M);
  from_gpu(hz_ctrl_h.data(), hz_ctrl_d, B * b);
  
  // Check values
  for(int i = 0; i < B * M; i++) {
    assert(std::abs(hz_mode_h[i] - alpha * 0.5f) < 1e-5f);
  }
  for(int i = 0; i < B * b; i++) {
    assert(std::abs(hz_ctrl_h[i] - rho * 0.3f) < 1e-5f);
  }
  
  cudaFree(lambda_m_d);
  cudaFree(c_bits_d);
  cudaFree(hz_mode_d);
  cudaFree(hz_ctrl_d);
  
  std::cout << "  ✓ Cost coefficient builder tests passed" << std::endl;
}

// Test 3: Temporal chain penalty
void test_temporal_chain() {
  std::cout << "Testing temporal chain penalty..." << std::endl;

  int B = 2, b = 4;
  float eta = 1.0f;  // Use 1.0 for simpler testing
  
  // mem_t: [0,0,0,0], [1,1,1,1]
  // mem_tp: [0,0,0,0], [1,0,1,0]
  // Hamming distances: 0, 2
  std::vector<uint8_t> mem_t_h(B * b);
  std::vector<uint8_t> mem_tp_h(B * b);

  // Batch 0: all zeros
  for(int i = 0; i < b; i++) {
    mem_t_h[i] = 0;
    mem_tp_h[i] = 0;
  }

  // Batch 1: [1,1,1,1] -> [1,0,1,0]
  mem_t_h[b] = 1; mem_t_h[b+1] = 1; mem_t_h[b+2] = 1; mem_t_h[b+3] = 1;
  mem_tp_h[b] = 1; mem_tp_h[b+1] = 0; mem_tp_h[b+2] = 1; mem_tp_h[b+3] = 0;
  
  uint8_t* mem_t_d = gpu_alloc<uint8_t>(B * b);
  uint8_t* mem_tp_d = gpu_alloc<uint8_t>(B * b);
  float* pair_cost_d = gpu_alloc<float>(B);
  
  to_gpu(mem_t_d, mem_t_h.data(), B * b);
  to_gpu(mem_tp_d, mem_tp_h.data(), B * b);
  
  build_temporal_chain(mem_t_d, mem_tp_d, pair_cost_d, eta, B, b);
  
  std::vector<float> pair_cost_h(B);
  from_gpu(pair_cost_h.data(), pair_cost_d, B);

  std::cout << "    pair_cost[0] = " << pair_cost_h[0] << " (expected 0.0)" << std::endl;
  std::cout << "    pair_cost[1] = " << pair_cost_h[1] << " (expected " << (eta * 2.0f) << ")" << std::endl;

  assert(std::abs(pair_cost_h[0] - 0.0f) < 1e-4f);
  assert(std::abs(pair_cost_h[1] - (eta * 2.0f)) < 1e-4f);
  
  cudaFree(mem_t_d);
  cudaFree(mem_tp_d);
  cudaFree(pair_cost_d);
  
  std::cout << "  ✓ Temporal chain penalty tests passed" << std::endl;
}

// Test 4: Reward gradient
void test_reward_grad() {
  std::cout << "Testing reward gradient..." << std::endl;
  
  int B = 2, b = 4, vmax = 15;
  std::vector<int> val_h = {7, 15};  // center and max
  
  int* val_d = gpu_alloc<int>(B);
  float* cj_d = gpu_alloc<float>(B * b);
  
  to_gpu(val_d, val_h.data(), B);
  reward_grad_bits(val_d, cj_d, B, b, vmax);
  
  std::vector<float> cj_h(B * b);
  from_gpu(cj_h.data(), cj_d, B * b);
  
  // val=7 (center) should give ~0, val=15 (max) should give ~1
  float center_val = cj_h[0];
  float max_val = cj_h[b];
  
  assert(std::abs(center_val) < 0.1f);
  assert(max_val > 0.9f);
  
  cudaFree(val_d);
  cudaFree(cj_d);
  
  std::cout << "  ✓ Reward gradient tests passed" << std::endl;
}

int main() {
  std::cout << "\n╔════════════════════════════════════════╗\n"
            << "║   CCC CUDA Kernel Test Suite           ║\n"
            << "╚════════════════════════════════════════╝\n\n";
  
  try {
    test_gray_codes();
    test_cost_coeffs();
    test_temporal_chain();
    test_reward_grad();
    
    std::cout << "\n✅ All kernel tests passed!\n\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "❌ Test failed: " << e.what() << "\n\n";
    return 1;
  }
}

