cmake_minimum_required(VERSION 3.10)
project(EdgeDetection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(edge_detection edge_detection.cpp)
target_link_libraries(edge_detection ${OpenCV_LIBS})
