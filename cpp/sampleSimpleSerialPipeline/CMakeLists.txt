cmake_minimum_required(VERSION 2.8)
project(sampleSimpleSerialPipeline)

set(SOURCE_FILES sampleSimpleSerialPipeline.cpp)

add_link_options( -fuse-ld=gold  )
add_compile_options(-Wall)
add_compile_options(-ggdb3)
add_compile_options(-O3)
add_compile_options(-Werror)
add_compile_options(-Wno-sign-compare)
add_compile_options(-std=c++17)

include_directories(/usr/include/TopsInference /usr/include/dtu/libprofile/ /usr/include/dtu/)
link_directories(/usr/lib /usr/local/lib/)
link_libraries(TopsInference pthread dl) 
add_executable(sampleSimpleSerialPipeline ${SOURCE_FILES})
