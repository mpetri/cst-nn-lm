cmake_minimum_required(VERSION 2.8)
project(cst-nn-lm)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (UNIX)
   # C++11
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   # For hardware popcount and other special instructions
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
   # Extensive warnings
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-braces")
   # Silence a warning bug in Boost
   if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
   endif ()
   # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
   if (USE_SANITIZERS)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
   endif ()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb") # Add debug info anyway
endif()

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

add_executable(create_cst_nn_lm src/create_cst_nn_lm.cpp)
target_link_libraries(create_cst_nn_lm
  ${Boost_LIBRARIES}
  dynet
  divsufsort
  divsufsort64
  )

add_executable(parse_input_files src/parse_input_files.cpp)
target_link_libraries(parse_input_files
  ${Boost_LIBRARIES}
  dynet
  divsufsort
  divsufsort64
  )
