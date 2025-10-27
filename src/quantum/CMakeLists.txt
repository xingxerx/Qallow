# Quantum Module CMakeLists.txt
# Builds quantum computing libraries and executables

cmake_minimum_required(VERSION 3.15)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSON_C json-c)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${JSON_C_INCLUDE_DIRS}
)

# Source files
set(QUANTUM_CORE_SOURCES
    quantum_core.c
)

set(QUANTUM_ALGORITHM_SUITE_SOURCES
    quantum_algorithm_suite.c
)

set(CUDAQ_QUICKSTART_SOURCES
    ${CMAKE_SOURCE_DIR}/examples/cudaq_quickstart.c
)

# Build quantum core library
add_library(qallow_quantum_core STATIC ${QUANTUM_CORE_SOURCES})
target_link_libraries(qallow_quantum_core ${JSON_C_LIBRARIES} m)

# Build quantum algorithm suite library
add_library(qallow_quantum_algorithms STATIC ${QUANTUM_ALGORITHM_SUITE_SOURCES})
target_link_libraries(qallow_quantum_algorithms ${JSON_C_LIBRARIES} m)

# Build CUDA-Q quickstart executable
add_executable(cudaq_quickstart ${CUDAQ_QUICKSTART_SOURCES})
target_link_libraries(cudaq_quickstart ${JSON_C_LIBRARIES} m)

# Installation
install(TARGETS 
    qallow_quantum_core
    qallow_quantum_algorithms
    cudaq_quickstart
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Testing
enable_testing()

# Test quantum core
add_test(
    NAME test_quantum_core
    COMMAND qallow_quantum_core
)

# Test algorithm suite
add_test(
    NAME test_quantum_algorithms
    COMMAND qallow_quantum_algorithms
)

# Test CUDA-Q quickstart
add_test(
    NAME test_cudaq_quickstart
    COMMAND cudaq_quickstart
)

# Print build info
message(STATUS "Quantum Module Configuration:")
message(STATUS "  - Quantum Core Library: qallow_quantum_core")
message(STATUS "  - Quantum Algorithms Library: qallow_quantum_algorithms")
message(STATUS "  - CUDA-Q Quickstart Executable: cudaq_quickstart")
message(STATUS "  - JSON-C Support: ${JSON_C_FOUND}")

