cmake_minimum_required(VERSION 3.10)

project(sampleONNXResnet)

set (TOPS_LIBS TopsInference)
set (CMAKE_BUILD_TYPE DEBUG)
set (CMAKE_CXX_FLAGS "-std=c++17")

add_executable(resnet resnet.cpp
          ../utils/tops_utils.cpp)

target_include_directories(resnet
    PRIVATE
    /usr/include/TopsInference/)

target_link_libraries(resnet ${TOPS_LIBS} stdc++fs)
