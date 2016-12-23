project(experiments)
cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
add_definitions ("-Wall")

set(SRC "./src")

add_subdirectory(${SRC})
