cmake_minimum_required(VERSION 2.8)

project(ElasticDistortion)
add_executable(ElasticDistortion "main.cpp")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ElasticDistortion ${OpenCV_LIBS})
