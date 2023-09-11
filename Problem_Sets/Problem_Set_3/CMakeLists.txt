############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
# collect source files

# collect source files
project(hw3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(CUB_IGNORE_DEPRECATED_CPP_DIALECT)

# Adding Cuda libraries
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Adding OpenCV liabraies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

add_executable(HW3 ${HW3_files} ${hdr} ${cu})

# Linking the libraries
target_link_libraries( HW3 ${OpenCV_LIBS} )
