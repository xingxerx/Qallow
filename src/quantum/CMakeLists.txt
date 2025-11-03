# Quantum Module CMakeLists.txt
# Builds quantum computing libraries and executables

cmake_minimum_required(VERSION 3.15)

# Find required packages - try CMake config first, then pkg-config, then manual
find_package(json-c CONFIG QUIET)
if(json-c_FOUND)
    # json-c found via CMake config
else()
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(JSON_C QUIET json-c)
    endif()
    
    # If still not found, create a manual target pointing to common locations
    if(NOT JSON_C_FOUND AND NOT json-c_FOUND)
        # Try to find json-c in common local build paths
        if(EXISTS "$ENV{HOME}/json-c/build")
            # json-c uses include/<package> structure; we need both source and build dirs
            set(JSON_C_INCLUDE_DIRS "$ENV{HOME}/json-c" "$ENV{HOME}/json-c/build")
            set(JSON_C_LIBRARIES "$ENV{HOME}/json-c/build/libjson-c.a")
            set(JSON_C_FOUND TRUE)
            message(STATUS "Using local json-c build from $ENV{HOME}/json-c/build")
            message(STATUS "JSON_C Include dirs: ${JSON_C_INCLUDE_DIRS}")
            message(STATUS "JSON_C Libraries: ${JSON_C_LIBRARIES}")
        else()
            message(FATAL_ERROR "json-c not found. Please install it or build from source in $HOME/json-c")
        endif()
    endif()
endif()

# Create a unified json-c target
if(NOT TARGET json-c::json-c)
    add_library(json-c::json-c INTERFACE IMPORTED)
    if(JSON_C_FOUND)
        set_target_properties(json-c::json-c PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${JSON_C_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${JSON_C_LIBRARIES}"
        )
    elseif(json-c_FOUND)
        # Use the CONFIG target
        set_target_properties(json-c::json-c PROPERTIES
            INTERFACE_LINK_LIBRARIES "json-c"
        )
    endif()
endif()

set(QALLOW_QUANTUM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
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
target_include_directories(qallow_quantum_core
    PUBLIC ${QALLOW_QUANTUM_INCLUDE_DIRS} $ENV{HOME}/json-c $ENV{HOME}/json-c/build
)
target_link_libraries(qallow_quantum_core
    PUBLIC ${JSON_C_LIBRARIES}
    PRIVATE m
)

# Build quantum algorithm suite library
add_library(qallow_quantum_algorithms STATIC ${QUANTUM_ALGORITHM_SUITE_SOURCES})
target_include_directories(qallow_quantum_algorithms
    PUBLIC ${QALLOW_QUANTUM_INCLUDE_DIRS} $ENV{HOME}/json-c $ENV{HOME}/json-c/build
)
target_link_libraries(qallow_quantum_algorithms
    PUBLIC ${JSON_C_LIBRARIES}
    PRIVATE m
)

# Build CUDA-Q quickstart executable
add_executable(cudaq_quickstart ${CUDAQ_QUICKSTART_SOURCES})
target_include_directories(cudaq_quickstart
    PRIVATE ${QALLOW_QUANTUM_INCLUDE_DIRS}
)
target_link_libraries(cudaq_quickstart
    PRIVATE
        json-c::json-c
        m
)

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

# Test CUDA-Q quickstart (only executable test)
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
