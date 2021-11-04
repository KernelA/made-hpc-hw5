cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(mpi VERSION 0.0.1 LANGUAGES CXX)

add_subdirectory(${PROJECT_SOURCE_DIR}/${PROJECT_NAME})

enable_testing()