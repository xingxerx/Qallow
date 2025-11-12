
cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(default_build_type "Debug")
    message(STATUS "No build type selected, defaulting to ${default_build_type}")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

option(QALLOW_ENABLE_CUDA "Enable CUDA backend" ON)

if(QALLOW_ENABLE_CUDA)
    project(Qallow LANGUAGES C CXX CUDA)
else()
    project(Qallow LANGUAGES C CXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(QALLOW_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else()
        message(STATUS "CUDA compiler not found. Disabling CUDA targets.")
        set(QALLOW_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

find_package(Threads REQUIRED)

set(QALLOW_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(qallow_core INTERFACE)
target_include_directories(qallow_core INTERFACE
    ${QALLOW_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cjson
)

set(QALLOW_ALGORITHM_SOURCES
    algorithms/ethics_core.c
    algorithms/ethics_learn.c
    algorithms/ethics_bayes.c
    algorithms/ethics_feed.c
    algorithms/quantum_suite.c
    third_party/cjson/cJSON.c
)

add_library(qallow_algorithms STATIC ${QALLOW_ALGORITHM_SOURCES})
target_link_libraries(qallow_algorithms PUBLIC qallow_core m)
target_include_directories(qallow_algorithms PUBLIC
    ${QALLOW_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cjson
)

# Define runtime library early so it can be used by backend_cpu
set(QALLOW_RUNTIME_SOURCES
    src/runtime/logging.cpp
    src/runtime/profiling.cpp
    src/runtime/env.cpp
    runtime/meta_introspect.c
    runtime/dl_integration.cpp
    src/runtime/telemetry_outputs.c
)

add_library(qallow_runtime STATIC ${QALLOW_RUNTIME_SOURCES})
target_link_libraries(qallow_runtime
    PUBLIC
        qallow_core
)
set_target_properties(qallow_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(qallow_runtime PUBLIC ${QALLOW_INCLUDE_DIRS})

set(QALLOW_BACKEND_CPU_SOURCES
    backend/cpu/adaptive.c
    backend/cpu/chronometric.c
    backend/cpu/error_handler.c
    backend/cpu/ethics.c
    backend/cpu/goal_synthesizer.c
    backend/cpu/govern.c
    backend/cpu/ingest.c
    backend/cpu/multi_pocket.c
    backend/cpu/overlay.c
    backend/cpu/phase14_coherence.c
    backend/cpu/phase15_convergence.c
    backend/cpu/phase12_elasticity.c
    backend/cpu/phase13_harmonic.c
    backend/cpu/phase7_core.c
    backend/cpu/lattice_integration.c
    backend/cpu/pocket.c
    backend/cpu/pocket_dimension.c
    backend/cpu/ppai.c
    backend/cpu/qallow_kernel.c
    backend/cpu/qcp.c
    backend/cpu/self_reflection.c
    backend/cpu/semantic_memory.c
    backend/cpu/telemetry.c
    backend/cpu/telemetry_ffi.c
    backend/cpu/transfer_engine.c
    backend/cpu/verify.c
    src/quantum/entanglement_bridge.c
    src/qallow_phase13.c
    backend/cpu/phase_wrapper_generic.c
)

add_library(qallow_backend_cpu STATIC ${QALLOW_BACKEND_CPU_SOURCES})
target_link_libraries(qallow_backend_cpu
    PUBLIC
        qallow_core
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)
target_compile_definitions(qallow_backend_cpu PUBLIC QALLOW_PHASE13_EMBEDDED)
target_include_directories(qallow_backend_cpu PUBLIC ${QALLOW_INCLUDE_DIRS})

add_library(qallow_backend_neuro STATIC
    backend/neuro/spike_coherence.c
)
target_link_libraries(qallow_backend_neuro
    PUBLIC
        qallow_core
        qallow_runtime
)
target_include_directories(qallow_backend_neuro PUBLIC ${QALLOW_INCLUDE_DIRS})

add_library(qallow_temporal_memory STATIC src/temporal_memory.c)
target_link_libraries(qallow_temporal_memory PUBLIC qallow_core m)
target_include_directories(qallow_temporal_memory PUBLIC ${QALLOW_INCLUDE_DIRS})


if(QALLOW_ENABLE_CUDA)
    target_compile_definitions(qallow_backend_cpu PUBLIC QALLOW_ENABLE_CUDA)
    target_include_directories(qallow_backend_cpu PUBLIC
        $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()

if(QALLOW_ENABLE_CUDA)
    enable_language(CUDA)
    set(QALLOW_CUDA_SOURCES
        backend/cuda/photonic.cu
        backend/cuda/pocket.cu
        backend/cuda/multi_pocket.cu
        backend/cuda/ppai_kernels.cu
        backend/cuda/qcp_kernels.cu
        backend/cuda/quantum.cu
        backend/cuda/phase16_meta_introspect.cu
        backend/cuda/phase14_gain.cu
        backend/cuda/mind_kernels.cu
        runtime/cuda_parallel.cu
        src/quantum/entanglement_bridge.c
    )
    add_library(qallow_backend_cuda STATIC ${QALLOW_CUDA_SOURCES})
    target_link_libraries(qallow_backend_cuda PUBLIC qallow_core)
    target_compile_definitions(qallow_backend_cuda PUBLIC QALLOW_WITH_CUDA)
    target_include_directories(qallow_backend_cuda
        PUBLIC ${QALLOW_INCLUDE_DIRS}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(qallow_backend_cuda PUBLIC CUDA::cudart)
    set_target_properties(qallow_backend_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Prevent GCC/Clang flags from being passed to NVCC
    # NVCC doesn't support -Wall, -Wextra, and other GCC-specific flags
    target_compile_options(qallow_backend_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

    add_custom_target(cudaq DEPENDS qallow_backend_cuda)
endif()

set(QALLOW_INTERFACE_SOURCES
    interface/main.c
    interface/launcher.c
    src/mind/registry.c
    src/mind/attention.c
    src/mind/memory.c
    src/mind/quantum_bridge.c
    src/mind/optimizer.c
    src/cli/bench_cmd.c
    src/cli/mind_cmd.c
    src/cli/entangle_cmd.c
    src/distributed/federated_learn.c
    src/ethics/multi_stakeholder.c
    src/runtime/telemetry_outputs.c
)

set(QALLOW_INTERFACE_BASE_SOURCES
    src/mind/registry.c
    src/mind/attention.c
    src/mind/memory.c
    src/mind/quantum_bridge.c
    src/mind/optimizer.c
    src/distributed/federated_learn.c
    src/ethics/multi_stakeholder.c
    src/runtime/telemetry_outputs.c
)

add_library(qallow_interface_base STATIC ${QALLOW_INTERFACE_BASE_SOURCES})
target_link_libraries(qallow_interface_base PUBLIC qallow_core)
target_include_directories(qallow_interface_base PUBLIC ${QALLOW_INCLUDE_DIRS})


# Main qallow executable
add_executable(qallow ${QALLOW_INTERFACE_SOURCES})
set_source_files_properties(${QALLOW_INTERFACE_SOURCES} PROPERTIES LANGUAGE C)
target_link_libraries(qallow
    PRIVATE
        qallow_interface_base
        qallow_backend_cpu
        qallow_backend_neuro
        qallow_temporal_memory
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)

if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow PRIVATE QALLOW_ENABLE_CUDA)
endif()

# Unified runner executable
add_executable(qallow_unified ${QALLOW_INTERFACE_SOURCES})
set_source_files_properties(${QALLOW_INTERFACE_SOURCES} PROPERTIES LANGUAGE C)
target_link_libraries(qallow_unified
    PRIVATE
        qallow_interface_base
        qallow_backend_cpu
        qallow_backend_neuro
        qallow_algorithms
        qallow_runtime
        qallow_temporal_memory
        Threads::Threads
        m
)
target_compile_definitions(qallow_unified PRIVATE QALLOW_UNIFIED)

if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_unified PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow_unified PRIVATE QALLOW_ENABLE_CUDA)
    set_target_properties(qallow_unified PROPERTIES OUTPUT_NAME "qallow_unified_cuda")
else()
    add_library(qallow_backend_cuda_fallback STATIC
        backend/cpu/fallback/photonic_fallback.c
        backend/cpu/fallback/quantum_fallback.c
        backend/cpu/fallback/phase16_meta_introspect_fallback.c
        backend/cpu/fallback/phase14_gain_fallback.c
    )
    target_link_libraries(qallow_backend_cuda_fallback PUBLIC qallow_core)
    target_include_directories(qallow_backend_cuda_fallback PUBLIC ${QALLOW_INCLUDE_DIRS})
    target_link_libraries(qallow_unified PRIVATE qallow_backend_cuda_fallback)
    set_target_properties(qallow_unified PROPERTIES OUTPUT_NAME "qallow_unified_cpu")
endif()

# Optional SDL-based telemetry UI
find_package(SDL2 QUIET)
find_package(SDL2_ttf QUIET)

if(SDL2_FOUND AND NOT TARGET SDL2::SDL2)
    add_library(SDL2::SDL2 INTERFACE IMPORTED)
    if(DEFINED SDL2_INCLUDE_DIRS AND SDL2_INCLUDE_DIRS)
        set_property(TARGET SDL2::SDL2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIRS}")
    elseif(DEFINED SDL2_INCLUDE_DIR AND SDL2_INCLUDE_DIR)
        set_property(TARGET SDL2::SDL2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIR}")
    endif()
    if(DEFINED SDL2_LIBRARIES AND SDL2_LIBRARIES)
        set_property(TARGET SDL2::SDL2 PROPERTY INTERFACE_LINK_LIBRARIES "${SDL2_LIBRARIES}")
    elseif(DEFINED SDL2_LIBRARY AND SDL2_LIBRARY)
        set_property(TARGET SDL2::SDL2 PROPERTY INTERFACE_LINK_LIBRARIES "${SDL2_LIBRARY}")
    endif()
endif()

if(SDL2_ttf_FOUND AND NOT TARGET SDL2_ttf::SDL2_ttf)
    add_library(SDL2_ttf::SDL2_ttf INTERFACE IMPORTED)
    if(DEFINED SDL2_TTF_INCLUDE_DIRS)
        set_property(TARGET SDL2_ttf::SDL2_ttf PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SDL2_TTF_INCLUDE_DIRS}")
    elseif(DEFINED SDL2_TTF_INCLUDE_DIR)
        set_property(TARGET SDL2_ttf::SDL2_ttf PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${SDL2_TTF_INCLUDE_DIR}")
    endif()
    if(DEFINED SDL2_TTF_LIBRARIES AND SDL2_TTF_LIBRARIES)
        set_property(TARGET SDL2_ttf::SDL2_ttf PROPERTY INTERFACE_LINK_LIBRARIES "${SDL2_TTF_LIBRARIES}")
    elseif(DEFINED SDL2_TTF_LIBRARY AND SDL2_TTF_LIBRARY)
        set_property(TARGET SDL2_ttf::SDL2_ttf PROPERTY INTERFACE_LINK_LIBRARIES "${SDL2_TTF_LIBRARY}")
    endif()
endif()

if(SDL2_FOUND AND SDL2_ttf_FOUND AND TARGET SDL2::SDL2 AND TARGET SDL2_ttf::SDL2_ttf)
    add_executable(qallow_ui
        interface/qallow_ui.c
    )
    target_link_libraries(qallow_ui
        PRIVATE
            SDL2::SDL2
            SDL2_ttf::SDL2_ttf
    )
    target_compile_definitions(qallow_ui PRIVATE SDL_MAIN_HANDLED)
    message(STATUS "SDL2 + SDL2_ttf detected; qallow_ui target enabled.")
else()
    message(STATUS "SDL2 + SDL2_ttf not detected; qallow_ui target disabled.")
endif()

set(_qallow_install_targets)
if(TARGET qallow)
    list(APPEND _qallow_install_targets qallow)
endif()
if(TARGET qallow_unified)
    list(APPEND _qallow_install_targets qallow_unified)
endif()

if(_qallow_install_targets)
    install(TARGETS ${_qallow_install_targets}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Examples and benchmarks
add_executable(qallow_throughput_bench
    examples/benchmarks/throughput_bench.c
    interface/phase_runners.c
)
target_include_directories(qallow_throughput_bench PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${QALLOW_INCLUDE_DIRS}
)
target_link_libraries(qallow_throughput_bench
    PRIVATE
        qallow_interface_base
        qallow_backend_cpu
        qallow_backend_neuro
        qallow_temporal_memory
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)
if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_throughput_bench PRIVATE qallow_backend_cuda CUDA::cudart)
endif()
# Phase demos also have linker issues - disabled
# if(QALLOW_ENABLE_CUDA)
#     set(PHASE_DEMOS 01 02 03 04 05 06 07 08 09 10 11 12 13)
#     foreach(phase IN LISTS PHASE_DEMOS)
#         set(target_name "phase${phase}_demo")
#         add_executable(${target_name} examples/phase_demos/phase${phase}_demo.cu)
#     endforeach()
# endif()

add_custom_target(qallow_examples)
# Always build alg_ccc (it provides CPU stub if CUDA is disabled)
add_subdirectory(alg_ccc)

# Add optional quantum module if the directory has a CMakeLists.txt
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/quantum/CMakeLists.txt)
    add_subdirectory(src/quantum)
else()
    message(STATUS "Skipping quantum module (optional dependency)")
endif()

enable_testing()

add_executable(qallow_unit_ethics tests/unit/test_ethics_core.c)
target_link_libraries(qallow_unit_ethics PRIVATE qallow_algorithms m)
add_test(NAME unit_ethics_core COMMAND qallow_unit_ethics)

add_executable(qallow_unit_dl_integration tests/unit/test_dl_integration.c)
target_link_libraries(qallow_unit_dl_integration PRIVATE qallow_runtime m)
add_test(NAME unit_dl_integration COMMAND qallow_unit_dl_integration)

if(QALLOW_ENABLE_CUDA)
    add_executable(qallow_unit_cuda_parallel tests/unit/test_cuda_parallel.cu)
    target_link_libraries(qallow_unit_cuda_parallel PRIVATE qallow_backend_cuda CUDA::cudart m)
    add_test(NAME unit_cuda_parallel COMMAND qallow_unit_cuda_parallel)
endif()

add_executable(qallow_test_temporal_memory tests/test_temporal_memory.c src/runtime/temporal_memory.c)
target_include_directories(qallow_test_temporal_memory PRIVATE ${QALLOW_INCLUDE_DIRS})
target_link_libraries(qallow_test_temporal_memory PRIVATE m)
add_test(NAME test_temporal_memory COMMAND qallow_test_temporal_memory)

# TODO: Fix integration_smoke linker errors
# add_executable(qallow_integration_smoke
#     tests/integration/test_vm_smoke.c
#     interface/main.c
# )
# set_source_files_properties(
#     tests/integration/test_vm_smoke.c
#     interface/main.c
#     PROPERTIES LANGUAGE CXX
# )
# set_target_properties(qallow_integration_smoke PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(qallow_integration_smoke
#     PRIVATE
#         -Wl,--whole-archive
#         qallow_interface_base
#         qallow_backend_cpu
#         qallow_backend_neuro
#         qallow_algorithms
#         qallow_runtime
#         -Wl,--no-whole-archive
#         Threads::Threads
#         m
# )
# 
# if(QALLOW_ENABLE_CUDA)
#     target_link_libraries(qallow_integration_smoke PRIVATE qallow_backend_cuda CUDA::cudart)
#     target_compile_definitions(qallow_integration_smoke PRIVATE QALLOW_ENABLE_CUDA)
# endif()
# 
# add_test(NAME integration_vm COMMAND qallow_integration_smoke)
