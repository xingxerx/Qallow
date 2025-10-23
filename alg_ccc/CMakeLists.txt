cmake_minimum_required(VERSION 3.22)
project(alg_ccc LANGUAGES CXX CUDA)

add_library(alg_ccc STATIC
    hamiltonian.cu
    koopman_cuda.cu
)

target_include_directories(alg_ccc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(alg_ccc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(alg_ccc PUBLIC cxx_std_20)
target_link_libraries(alg_ccc PUBLIC cuda cudart)

# Optional: small CLI stub (built only if top-level defines TARGET qallow_unified_cuda)
if(TARGET qallow_unified_cuda)
  target_link_libraries(qallow_unified_cuda PUBLIC alg_ccc)
endif()
