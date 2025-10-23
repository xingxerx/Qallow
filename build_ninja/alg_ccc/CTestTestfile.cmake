# CMake generated Testfile for 
# Source directory: /root/Qallow/alg_ccc
# Build directory: /root/Qallow/build_ninja/alg_ccc
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[GrayCodeTest]=] "/root/Qallow/build_ninja/alg_ccc/test_gray")
set_tests_properties([=[GrayCodeTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/alg_ccc/CMakeLists.txt;24;add_test;/root/Qallow/alg_ccc/CMakeLists.txt;0;")
add_test([=[KernelTests]=] "/root/Qallow/build_ninja/alg_ccc/test_kernels")
set_tests_properties([=[KernelTests]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/alg_ccc/CMakeLists.txt;31;add_test;/root/Qallow/alg_ccc/CMakeLists.txt;0;")
add_test([=[alg_ccc_test_gray]=] "/root/Qallow/build_ninja/alg_ccc/alg_ccc_test_gray")
set_tests_properties([=[alg_ccc_test_gray]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/alg_ccc/CMakeLists.txt;37;add_test;/root/Qallow/alg_ccc/CMakeLists.txt;0;")
