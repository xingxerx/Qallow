cmake_minimum_required(VERSION 3.10)
project(qallow-alg)

# Python executable wrapper for ALG
# This creates a shell script that invokes the Python CLI

set(ALG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ALG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")

# Create wrapper script
configure_file(
    "${ALG_SOURCE_DIR}/alg.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/alg"
    @ONLY
)

# Install wrapper script
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/alg"
    DESTINATION "${ALG_INSTALL_DIR}"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# Install Python modules
install(
    DIRECTORY "${ALG_SOURCE_DIR}/core"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/qallow/alg"
)

install(
    FILES "${ALG_SOURCE_DIR}/main.py"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/qallow/alg"
    PERMISSIONS OWNER_READ OWNER_WRITE
                GROUP_READ
                WORLD_READ
)

install(
    FILES "${ALG_SOURCE_DIR}/qaoa_spsa.py"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/qallow/alg"
    PERMISSIONS OWNER_READ OWNER_WRITE
                GROUP_READ
                WORLD_READ
)

# Install documentation
install(
    FILES "${ALG_SOURCE_DIR}/README.md"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/qallow/alg"
)

message(STATUS "ALG - Quantum Algorithm Optimizer configured")
message(STATUS "  Install directory: ${ALG_INSTALL_DIR}")
message(STATUS "  Python modules: ${CMAKE_INSTALL_PREFIX}/lib/qallow/alg")

