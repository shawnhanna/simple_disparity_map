cmake_minimum_required(VERSION 2.8)

project(simple_disparity_map)

find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set (OpenMP_FLAGS "${OpenMP_CXX_FLAGS}")
  set (OpenMP_LIBS gomp)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Create the disparity_map library
add_library(disparity_map SimpleDisparity.cpp)
target_link_libraries( disparity_map ${OpenCV_LIBS} ${OpenMP_LIBS})
target_compile_options(disparity_map PRIVATE ${OpenMP_FLAGS})

# add the executable
add_executable(simple_disparity_map simple_disparity_map.cpp)

target_link_libraries( simple_disparity_map disparity_map ${OpenCV_LIBS} )
