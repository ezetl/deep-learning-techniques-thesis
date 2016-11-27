project(experiments)
cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
add_definitions ("-Wall")

    
set(COM "./common")
set(MNI "./mnist")
set(ALE "./alexnet")
set(KIT "./kitti")
set(SCE "./scene_recognition")
set(SF "./sf")


# TODO: change this once the filenames are definitive
file(GLOB INFILES "${MNI}/*pp" "${ALE}/*pp" "${KIT}/*pp" "${SCE}/*pp" "${SF}/*pp")

macro(add_code fullname)
   include_directories(${COM})
   set(EXECUTABLE_OUTPUT_PATH "${experiments_BINARY_DIR}/bin" CACHE INTERNAL "")
   set(CMAKE_INCLUDE_CURRENT_DIR ON)
   set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

   get_filename_component(name ${fullname} NAME)
   get_filename_component(ext ${fullname} EXT)

   STRING(REGEX REPLACE ${ext} "" outname "${name}")

   # OpenCV
   find_package(OpenCV)
   if (OpenCV_FOUND)
       include_directories(${OpenCV_INCLUDE_DIRS})
   endif()

   # Caffe
   find_package(Caffe)
   include_directories(${Caffe_INCLUDE_DIRS})
   add_definitions(${Caffe_DEFINITIONS})

   add_executable(${outname} ${fullname})

   target_link_libraries(${outname} ${Caffe_LIBRARIES} ${OpenCV_LIBS})
endmacro()


FOREACH(infile ${INFILES})
    message(STATUS "Processing file: ${infile}")
    add_code(${infile})
endforeach(infile)
