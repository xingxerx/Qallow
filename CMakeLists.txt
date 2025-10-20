cmake_minimum_required(VERSION 3.20)

project(Qallow LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QALLOW_ENABLE_CUDA "Enable CUDA backend" ON)

if(QALLOW_ENABLE_CUDA AND NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler not found. Disabling CUDA targets.")
    set(QALLOW_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
endif()

if(QALLOW_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)

FetchContent_MakeAvailable(spdlog)

set(QALLOW_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

add_library(qallow_core INTERFACE)
target_include_directories(qallow_core INTERFACE ${QALLOW_INCLUDE_DIRS})

set(QALLOW_ALGORITHM_SOURCES
    algorithms/ethics_core.c
    algorithms/ethics_learn.c
    algorithms/ethics_bayes.c
    algorithms/ethics_feed.c
)

add_library(qallow_algorithms STATIC ${QALLOW_ALGORITHM_SOURCES})
target_link_libraries(qallow_algorithms PUBLIC qallow_core m)
target_include_directories(qallow_algorithms PUBLIC ${QALLOW_INCLUDE_DIRS})

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
    backend/cpu/transfer_engine.c
    backend/cpu/verify.c
    src/qallow_phase13.c
)

add_library(qallow_backend_cpu STATIC ${QALLOW_BACKEND_CPU_SOURCES})
target_link_libraries(qallow_backend_cpu
    PUBLIC
        qallow_core
        qallow_algorithms
        Threads::Threads
        m
)
target_compile_definitions(qallow_backend_cpu PUBLIC QALLOW_PHASE13_EMBEDDED)
target_include_directories(qallow_backend_cpu PUBLIC ${QALLOW_INCLUDE_DIRS})

if(QALLOW_ENABLE_CUDA)
    target_compile_definitions(qallow_backend_cpu PUBLIC QALLOW_ENABLE_CUDA)
    target_include_directories(qallow_backend_cpu PUBLIC
        $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()

set(QALLOW_RUNTIME_SOURCES
    src/runtime/logging.cpp
    src/runtime/profiling.cpp
    src/runtime/env.cpp
    runtime/meta_introspect.c
)

add_library(qallow_runtime STATIC ${QALLOW_RUNTIME_SOURCES})
target_link_libraries(qallow_runtime
    PUBLIC
        qallow_core
        spdlog::spdlog
)
target_compile_features(qallow_runtime PUBLIC cxx_std_17)
target_include_directories(qallow_runtime PUBLIC ${QALLOW_INCLUDE_DIRS})

if(QALLOW_ENABLE_CUDA)
    enable_language(CUDA)
    set(QALLOW_CUDA_SOURCES
        backend/cuda/photonic.cu
        backend/cuda/pocket.cu
        backend/cuda/ppai_kernels.cu
        backend/cuda/qcp_kernels.cu
        backend/cuda/quantum.cu
        backend/cuda/phase16_meta_introspect.cu
    )
    add_library(qallow_backend_cuda STATIC ${QALLOW_CUDA_SOURCES})
    target_link_libraries(qallow_backend_cuda PUBLIC qallow_core)
    target_compile_definitions(qallow_backend_cuda PUBLIC QALLOW_WITH_CUDA)
    target_include_directories(qallow_backend_cuda PUBLIC ${QALLOW_INCLUDE_DIRS})
    find_package(CUDAToolkit REQUIRED)
endif()

set(QALLOW_INTERFACE_SOURCES
    interface/main.c
    interface/launcher.c
)

add_executable(qallow ${QALLOW_INTERFACE_SOURCES})
target_link_libraries(qallow
    PRIVATE
        qallow_backend_cpu
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)

if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow PRIVATE QALLOW_ENABLE_CUDA)
endif()

add_executable(qallow_unified ${QALLOW_INTERFACE_SOURCES})
target_link_libraries(qallow_unified
    PRIVATE
        qallow_backend_cpu
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)
target_compile_definitions(qallow_unified PRIVATE QALLOW_UNIFIED)

if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_unified PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow_unified PRIVATE QALLOW_ENABLE_CUDA)
endif()

install(TARGETS qallow qallow_unified
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Examples and benchmarks
add_executable(qallow_throughput_bench
    examples/benchmarks/throughput_bench.c
    interface/main.c
)
target_link_libraries(qallow_throughput_bench
    PRIVATE
        qallow_runtime
        qallow_algorithms
        qallow_backend_cpu
        Threads::Threads
        m
)
if(QALLOW_ENABLE_CUDA)
    target_compile_definitions(qallow_throughput_bench PRIVATE QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_throughput_bench PRIVATE qallow_backend_cuda CUDA::cudart)
endif()

if(QALLOW_ENABLE_CUDA)
    set(PHASE_DEMOS
        01 02 03 04 05 06 07 08 09 10 11 12 13
    )
    foreach(phase IN LISTS PHASE_DEMOS)
        set(target_name "phase${phase}_demo")
        add_executable(${target_name} examples/phase_demos/phase${phase}_demo.cu)
        target_link_libraries(${target_name}
            PRIVATE
                qallow_runtime
                CUDA::cudart
        )
        target_compile_definitions(${target_name} PRIVATE QALLOW_ENABLE_CUDA)
    endforeach()
endif()

add_custom_target(qallow_examples DEPENDS qallow_throughput_bench)
if(QALLOW_ENABLE_CUDA)
    foreach(phase IN LISTS PHASE_DEMOS)
        add_dependencies(qallow_examples phase${phase}_demo)
    endforeach()
endif()

enable_testing()

add_executable(qallow_unit_ethics tests/unit/test_ethics_core.c)
target_link_libraries(qallow_unit_ethics PRIVATE qallow_algorithms m)
add_test(NAME unit_ethics_core COMMAND qallow_unit_ethics)

add_executable(qallow_integration_smoke
    tests/integration/test_vm_smoke.c
    interface/main.c
)
target_link_libraries(qallow_integration_smoke
    PRIVATE
        qallow_backend_cpu
        qallow_algorithms
        qallow_runtime
        Threads::Threads
        m
)

if(QALLOW_ENABLE_CUDA)
    target_link_libraries(qallow_integration_smoke PRIVATE qallow_backend_cuda CUDA::cudart)
    target_compile_definitions(qallow_integration_smoke PRIVATE QALLOW_ENABLE_CUDA)
endif()

add_test(NAME integration_vm COMMAND qallow_integration_smoke)
