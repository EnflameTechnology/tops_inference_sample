cmake_minimum_required(VERSION 3.10)

project(SampleONNXYolov5)

set (TOPS_LIBS TopsInference)
set (CMAKE_BUILD_TYPE DEBUG)
set (CMAKE_CXX_FLAGS "-std=c++17 -O3")

add_executable(yolov5s yolov5.cpp
          ../utils/tops_utils.cpp)

target_include_directories(yolov5s
    PRIVATE
    /usr/include/TopsInference/)

target_link_libraries(yolov5s ${TOPS_LIBS} stdc++fs)
