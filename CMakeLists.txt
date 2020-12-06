cmake_minimum_required(VERSION 3.17)
project(Mixture_Guassion)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR E:\\opencv4.x\\opencv\\mingw_build\\install)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Mixture_Guassion main.cpp )


target_link_libraries(Mixture_Guassion ${OpenCV_LIBS})