cmake_minimum_required(VERSION 3.10)
project(PlaneEstimation)

find_package(Eigen3 3.3 REQUIRED)

add_executable(PlaneEstimation main.cpp)
target_link_libraries(PlaneEstimation Eigen3::Eigen)

