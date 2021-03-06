cmake_minimum_required(VERSION 2.8)
project(cst-nn-lm)

cmake_host_system_information(RESULT current_host_name QUERY HOSTNAME)
message("-- current_host_name: " ${current_host_name})
if("${current_host_name}" MATCHES "^slug$")
  message("ENABLE CUDA AND SET CUDA 8.0 DIR")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0/")
  set(BACKEND "cuda")
endif()
if("${current_host_name}" MATCHES "^spartan")
  message("-- ENABLE CUDA AND SET CUDA 9.0 DIR")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/easybuild/software/CUDA/9.1.85")
  set(CUDNN_ROOT "/usr/local/easybuild/software/cuDNN/7.1-GCC-6.2.0-CUDA-9.1.85")
  set(BACKEND "cuda")
endif()

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# C++ 14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# For hardware popcount and other special instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# Extensive warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-missing-braces -Wno-unused-variable")
# Silence a warning bug in Boost
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb") # Add debug info anyway

# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS iostreams unit_test_framework filesystem system log log_setup date_time chrono program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(/opt/intel/mkl/lib/intel64)
link_directories(/opt/intel/compilers_and_libraries/linux/lib/intel64/)
add_definitions(-DBOOST_LOG_DYN_LINK)

add_subdirectory(external/eigen-git-mirror EXCLUDE_FROM_ALL)
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/eigen-git-mirror)
add_subdirectory(external/dynet EXCLUDE_FROM_ALL)
include_directories(${CMAKE_SOURCE_DIR}/external/dynet/)

add_subdirectory(external/sdsl-lite EXCLUDE_FROM_ALL)
include_directories(${CMAKE_BINARY_DIR}/external/sdsl-lite/include/)
include_directories(${CMAKE_BINARY_DIR}/external/sdsl-lite/external/libdivsufsort/include/)

#
include_directories("./include/")

add_executable(create_lm src/create_lm.cpp)
target_link_libraries(create_lm
  ${Boost_LIBRARIES}
  dynet
  divsufsort
  divsufsort64
  )


add_executable(create_parsed src/create_parsed.cpp)
target_link_libraries(create_parsed
  ${Boost_LIBRARIES}
  dynet
  divsufsort
  divsufsort64
  )

  add_executable(create_lm_ngram src/create_lm_ngram.cpp)
  target_link_libraries(create_lm_ngram
    ${Boost_LIBRARIES}
    dynet
    divsufsort
    divsufsort64
    )