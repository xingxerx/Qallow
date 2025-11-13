
cmake_minimum_required(VERSION 3.22)

if(QALLOW_ENABLE_CUDA)
  project(alg_ccc LANGUAGES CXX CUDA)
  add_library(alg_ccc STATIC
    hamiltonian.cu
    koopman_cuda.cu
    gray.cpp
  )
  set_target_properties(alg_ccc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(alg_ccc PUBLIC cuda cudart)
  set_target_properties(alg_ccc PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
else()
  project(alg_ccc LANGUAGES CXX)
  add_library(alg_ccc STATIC
    gray.cpp
  )
  set_target_properties(alg_ccc PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(alg_ccc PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
endif()

target_include_directories(alg_ccc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional: small CLI stub (built only if top-level defines TARGET qallow_unified_cuda)
if(TARGET qallow_unified_cuda)
  target_link_libraries(qallow_unified_cuda PUBLIC alg_ccc)
endif()

# Unit tests
enable_testing()
add_executable(test_gray tests/test_gray.cpp)
target_link_libraries(test_gray PRIVATE alg_ccc)
target_include_directories(test_gray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME GrayCodeTest COMMAND test_gray)

if(QALLOW_ENABLE_CUDA)
  add_executable(test_kernels tests/test_kernels.cu)
  target_link_libraries(test_kernels PRIVATE alg_ccc)
  target_include_directories(test_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(test_kernels PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  set_target_properties(test_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME KernelTests COMMAND test_kernels)
endif()

# Tests
add_executable(alg_ccc_test_gray tests/test_gray.cpp)
target_include_directories(alg_ccc_test_gray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alg_ccc_test_gray PRIVATE alg_ccc)
add_test(NAME alg_ccc_test_gray COMMAND alg_ccc_test_gray)
