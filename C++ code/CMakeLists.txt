cmake_minimum_required(VERSION 3.10)
project(EdgeDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(edge_detection main.cpp EdgeDetector.cpp)
add_executable(edge_detection test.cpp)
target_link_libraries(edge_detection ${OpenCV_LIBS})
