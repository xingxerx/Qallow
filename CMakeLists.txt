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

# Backend sources
file(GLOB CPU_BACKEND_SOURCES "backend/cpu/*.c")

if(QALLOW_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA backend is enabled.")
    file(GLOB CUDA_BACKEND_SOURCES "backend/cuda/*.cu")
    
    # Add sources for the CUDA build
    add_executable(qallow ${INTERFACE_SOURCES} ${CPU_BACKEND_SOURCES} ${CUDA_BACKEND_SOURCES})
    
    # Link CUDA libraries
    target_link_libraries(qallow PRIVATE ${CMAKE_CUDA_LIBRARIES})
    
    # Add a compile definition to indicate CUDA is used
    target_compile_definitions(qallow PRIVATE USE_CUDA)
    
else()
    message(STATUS "CUDA backend is disabled or not available. Building CPU-only version.")
    
    # Add sources for the CPU-only build
    add_executable(qallow ${INTERFACE_SOURCES} ${CPU_BACKEND_SOURCES})
    
endif()

# --- Installation ---
install(TARGETS qallow DESTINATION bin)

# --- Build Scripts ---
# The user can still use the scripts in the scripts/ folder for convenience
# For example, a build.bat could now run:
# mkdir build
# cd build
# cmake ..
# cmake --build .
