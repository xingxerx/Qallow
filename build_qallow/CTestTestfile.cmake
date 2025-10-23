# CMake generated Testfile for 
# Source directory: /root/Qallow
# Build directory: /root/Qallow/build_qallow
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[unit_ethics_core]=] "/root/Qallow/build_qallow/qallow_unit_ethics")
set_tests_properties([=[unit_ethics_core]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/CMakeLists.txt;311;add_test;/root/Qallow/CMakeLists.txt;0;")
add_test([=[unit_dl_integration]=] "/root/Qallow/build_qallow/qallow_unit_dl_integration")
set_tests_properties([=[unit_dl_integration]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/CMakeLists.txt;315;add_test;/root/Qallow/CMakeLists.txt;0;")
add_test([=[unit_cuda_parallel]=] "/root/Qallow/build_qallow/qallow_unit_cuda_parallel")
set_tests_properties([=[unit_cuda_parallel]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/CMakeLists.txt;320;add_test;/root/Qallow/CMakeLists.txt;0;")
add_test([=[integration_vm]=] "/root/Qallow/build_qallow/qallow_integration_smoke")
set_tests_properties([=[integration_vm]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/Qallow/CMakeLists.txt;342;add_test;/root/Qallow/CMakeLists.txt;0;")
subdirs("_deps/spdlog-build")
subdirs("alg_ccc")
