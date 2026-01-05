cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sampleBert)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED)
if (Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

file(GLOB SRC_FILE ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB UTILS_FILE ${PROJECT_SOURCE_DIR}/utils/*.cpp ${PROJECT_SOURCE_DIR}/utils/*.hpp ../utils/tops_utils.cpp)

add_executable(${PROJECT_NAME} ${UTILS_FILE} ${SRC_FILE})

target_link_libraries(${PROJECT_NAME} TopsInference)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} stdc++fs)
target_include_directories(${PROJECT_NAME} PUBLIC "utils" /opt/tops/include)
