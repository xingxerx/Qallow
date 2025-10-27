I have identified and fixed a CMake configuration issue. The build was failing due to a mismatch between the CMake generator specified in the configuration (Ninja) and the one used to initialize the build directory (Unix Makefiles).

I have corrected this by editing the `CMakeCache.txt` file in the `build` directory to set the `CMAKE_GENERATOR` to `Unix Makefiles`.

The build should now proceed without error the next time the CMake configuration is run.