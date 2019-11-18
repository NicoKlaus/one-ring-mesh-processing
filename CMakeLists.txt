cmake_minimum_required(VERSION 3.14)

set (CMAKE_CXX_STANDARD 11)

project(onering)


set(VTKm_ENABLE_CUDA ON CACHE BOOL "")
add_subdirectory(lib/vtk-m-v1.5.0)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/*")
file(GLOB source "${PROJECT_SOURCE_DIR}/src/*")

add_executable(onering source headers)
