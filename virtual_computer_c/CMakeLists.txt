cmake_minimum_required(VERSION 3.10)
project(VirtualComputer C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -fPIC")

# Math library for sin, cos, exp, etc.
find_library(MATH_LIBRARY m)

# Source files (with full paths)
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/neuromorphic_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/photonic_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_computer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)

# Headers
set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_simulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/neuromorphic_simulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/photonic_simulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_computer.h
)

# Create executable
add_executable(virtual_computer_demo ${SOURCES} ${HEADERS})

# Link math library
if(MATH_LIBRARY)
    target_link_libraries(virtual_computer_demo ${MATH_LIBRARY})
endif()

# Create library
add_library(virtual_computer_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/neuromorphic_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/photonic_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_computer.c
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

