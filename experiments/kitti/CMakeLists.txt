project(mnist.demo)
cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
add_definitions ("-Wall")

MACRO(add_code name)
   set(CMAKE_INCLUDE_CURRENT_DIR ON)

   set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

   # OpenCV
   find_package(OpenCV)
   if (OpenCV_FOUND)
       include_directories(${OpenCV_INCLUDE_DIRS})
   endif()

   # Caffe
   FIND_PACKAGE(Caffe)
   INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})
   ADD_DEFINITIONS(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

   ADD_EXECUTABLE(${name} ${name}.cpp)

   TARGET_LINK_LIBRARIES(${name}
                         ${Caffe_LIBRARIES}
                         ${OpenCV_LIBS})
ENDMACRO()

add_code(preprocess_kitti_egomotion)
