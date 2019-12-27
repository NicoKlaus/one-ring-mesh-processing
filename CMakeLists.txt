cmake_minimum_required(VERSION 3.16)

project(onering)
message("one ring test 0.1.0")
set (CMAKE_CXX_STANDARD 11)
set (BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/boost_1_72_0" CACHE PATH "")
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS ON )
set (Boost_NO_SYSTEM_PATHS  ON)
#LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/vtk-m-v1.5.0/build/lib/cmake/vtkm-1.5")
#message("modules ${CMAKE_MODULE_PATH}")

find_package(Boost 1.72.0 REQUIRED COMPONENTS program_options thread)


#set(VTKm_ENABLE_CUDA OFF CACHE BOOL "")
add_subdirectory(lib/vtk-m-v1.5.0)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/*")
file(GLOB source "${PROJECT_SOURCE_DIR}/src/*")
list(APPEND headers "${CMAKE_CURRENT_SOURCE_DIR}/lib/happly/happly.h")

add_executable(onering ${source} ${headers})
target_link_libraries(onering vtkm_cont Boost::program_options)
target_include_directories(onering PUBLIC 
		"${CMAKE_CURRENT_SOURCE_DIR}/include" 
		"${CMAKE_CURRENT_SOURCE_DIR}/lib/happly" 
		${Boost_INCLUDE_DIR})