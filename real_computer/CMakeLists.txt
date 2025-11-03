cmake_minimum_required(VERSION 3.18)
project(QallowRealComputer CUDA C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -fPIC")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_70")

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find Python for Cirq integration
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(REAL_CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/real_cuda.c
)

set(CIRQ_QUANTUM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cirq_quantum.c
)

set(REAL_COMPUTER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/real_computer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/real_cuda.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cirq_quantum.c
)

# Create library for real CUDA GPU interface
add_library(real_cuda STATIC ${REAL_CUDA_SOURCES})
target_link_libraries(real_cuda PUBLIC ${CUDA_LIBRARIES} m)

# Create library for Cirq quantum interface
add_library(cirq_quantum STATIC ${CIRQ_QUANTUM_SOURCES})
target_link_libraries(cirq_quantum PUBLIC ${Python3_LIBRARIES} m)

# Create library for real computer orchestrator
add_library(real_computer STATIC ${REAL_COMPUTER_SOURCES})
target_link_libraries(real_computer PUBLIC real_cuda cirq_quantum ${CUDA_LIBRARIES} ${Python3_LIBRARIES} m)

# Create main demo executable
add_executable(real_computer_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.c)
target_link_libraries(real_computer_demo PRIVATE real_computer)

# Installation targets
install(TARGETS real_cuda real_quantum real_computer real_computer_demo
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES real_cuda.h cirq_quantum.h real_computer.h
        DESTINATION include/qallow_real_computer)

# Print configuration info
message(STATUS "=== QallowRealComputer Build Configuration ===")
message(STATUS "CUDA Found: YES")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Python3 Found: YES")
message(STATUS "Python3 Version: ${Python3_VERSION}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "Build Real Computer GPU: YES")
message(STATUS "Build Real Quantum: YES")
