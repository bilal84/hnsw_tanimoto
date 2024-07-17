cmake_minimum_required(VERSION 3.4...3.18)
project(hnsw_tanimoto)

# Set the C++ standard to C++17 instead of C++14
set(CMAKE_CXX_STANDARD 17)

# Add the path to pybind11
set(CMAKE_PREFIX_PATH "/home/bilal/.local/lib/python3.10/site-packages/pybind11")

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the pybind11 module
pybind11_add_module(hnsw_tanimoto hnsw_tanimoto.cpp)