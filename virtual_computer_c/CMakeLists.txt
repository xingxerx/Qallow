cmake_minimum_required(VERSION 3.10)
project(VirtualComputer C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -fPIC")

# Math library for sin, cos, exp, etc.
find_library(MATH_LIBRARY m)

# Source files
set(SOURCES
    cuda_simulator.c
    neuromorphic_simulator.c
    photonic_simulator.c
    virtual_computer.c
    main.c
)

# Headers
set(HEADERS
    cuda_simulator.h
    neuromorphic_simulator.h
    photonic_simulator.h
    virtual_computer.h
)

# Create executable
add_executable(virtual_computer_demo ${SOURCES} ${HEADERS})

# Link math library
if(MATH_LIBRARY)
    target_link_libraries(virtual_computer_demo ${MATH_LIBRARY})
endif()

# Create library
add_library(virtual_computer_lib STATIC
    cuda_simulator.c
    neuromorphic_simulator.c
    photonic_simulator.c
    virtual_computer.c
)

if(MATH_LIBRARY)
    target_link_libraries(virtual_computer_lib ${MATH_LIBRARY})
endif()

# Installation
install(TARGETS virtual_computer_demo
        RUNTIME DESTINATION bin)

install(TARGETS virtual_computer_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES ${HEADERS}
        DESTINATION include/virtual_computer)

# Print configuration
message(STATUS "VirtualComputer Configuration:")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "  Math Library: ${MATH_LIBRARY}")
