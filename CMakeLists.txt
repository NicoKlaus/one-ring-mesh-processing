cmake_minimum_required(VERSION 3.16)

project(onering LANGUAGES CUDA CXX)
message("one ring test 0.1.3")
set (CMAKE_CXX_STANDARD 11)
set (BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/boost_1_72_0" CACHE PATH "")
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS ON )
set (Boost_NO_SYSTEM_PATHS  ON)

find_package(Boost 1.72.0 REQUIRED COMPONENTS program_options thread)
set(VTKm_ENABLE_TESTING OFF CACHE BOOL "")
add_subdirectory(lib/vtk-m-v1.5.0)
file(GLOB headers "${PROJECT_SOURCE_DIR}/include/*")
file(GLOB shared_source "${PROJECT_SOURCE_DIR}/src/shared/*")
file(GLOB onering_source "${PROJECT_SOURCE_DIR}/src/onering/*")
file(GLOB onering_vtkm_source "${PROJECT_SOURCE_DIR}/onering-vtkm/*")
list(APPEND headers "${CMAKE_CURRENT_SOURCE_DIR}/lib/happly/happly.h")

add_executable(onering ${onering_source} ${shared_source} ${headers})
target_link_libraries(onering Boost::program_options)
target_include_directories(onering PUBLIC 
		"${CMAKE_CURRENT_SOURCE_DIR}/include" 
		"${CMAKE_CURRENT_SOURCE_DIR}/lib/happly" 
		${Boost_INCLUDE_DIR})

add_executable(onering_vtkm ${onering_vtkm_source} ${shared_source} ${headers})
target_link_libraries(onering_vtkm Boost::program_options vtkm_cont vtkm_filter)
target_include_directories(onering_vtkm PUBLIC 
		"${CMAKE_CURRENT_SOURCE_DIR}/include" 
		"${CMAKE_CURRENT_SOURCE_DIR}/lib/happly" 
		${Boost_INCLUDE_DIR})