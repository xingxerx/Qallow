# CMakeLists.txt for Qallow

cmake_minimum_required(VERSION 3.20)
project(Qallow LANGUAGES C CUDA)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set CUDA standard if available
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# --- Options ---
option(QALLOW_ENABLE_CUDA "Enable CUDA backend" ON)

# --- Include Directories ---
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
)

# --- Source Files ---

# Core (shared logic, might be header-only for now)
# If there were .c files in core, they would be added here.

# Interface
file(GLOB INTERFACE_SOURCES "interface/*.c")
list(REMOVE_ITEM INTERFACE_SOURCES
     "${CMAKE_CURRENT_SOURCE_DIR}/interface/main_entry.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/interface/qallow_unified_main.c")

xlea# Unified main with ethics
set(UNIFIED_MAIN_SOURCE "interface/qallow_unified_main.c")

# Backend sources
file(GLOB CPU_BACKEND_SOURCES "backend/cpu/*.c")

set(ALGORITHMS_SOURCES
    algorithms/ethics_core.c
    algorithms/ethics_learn.c
    algorithms/ethics_bayes.c
    algorithms/ethics_feed.c
)

if(QALLOW_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA backend is enabled.")
    file(GLOB CUDA_BACKEND_SOURCES "backend/cuda/*.cu")
    
    # Build unified application with CUDA + Ethics
    add_executable(qallow_unified ${UNIFIED_MAIN_SOURCE} ${CPU_BACKEND_SOURCES} ${ALGORITHMS_SOURCES} ${CUDA_BACKEND_SOURCES})
    
    # Also build original for compatibility
    add_executable(qallow ${INTERFACE_SOURCES} ${CPU_BACKEND_SOURCES} ${ALGORITHMS_SOURCES} ${CUDA_BACKEND_SOURCES})
    
    # Link CUDA libraries
    target_link_libraries(qallow_unified PRIVATE ${CMAKE_CUDA_LIBRARIES} m)
    target_link_libraries(qallow PRIVATE ${CMAKE_CUDA_LIBRARIES} m)
    
    # Add compile definitions
    target_compile_definitions(qallow_unified PRIVATE USE_CUDA)
    target_compile_definitions(qallow PRIVATE USE_CUDA)
    
else()
    message(STATUS "CUDA backend is disabled or not available. Building CPU-only version.")
    
    # Build unified application CPU-only + Ethics
    add_executable(qallow_unified ${UNIFIED_MAIN_SOURCE} ${CPU_BACKEND_SOURCES} ${ALGORITHMS_SOURCES})
    
    # Also build original for compatibility
    add_executable(qallow ${INTERFACE_SOURCES} ${CPU_BACKEND_SOURCES} ${ALGORITHMS_SOURCES})
    
    # Link math library
    target_link_libraries(qallow_unified PRIVATE m)
    target_link_libraries(qallow PRIVATE m)
    
endif()

# --- Installation ---
install(TARGETS qallow qallow_unified DESTINATION bin)

# --- Build Scripts ---
# The user can still use the scripts in the scripts/ folder for convenience
# For example, a build.bat could now run:
# mkdir build
# cd build
# cmake ..
# cmake --build .
