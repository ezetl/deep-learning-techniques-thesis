set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++11")
add_definitions ("-Wall")

set(SRC ".")
#file(GLOB files "." "*.cpp")
set(files "${SRC}/preprocess_mnist_siamese.cpp" "${SRC}/preprocess_mnist_standar.cpp")

include_directories("${SRC}/utils")
include_directories("${SRC}/lmdb_creator")

add_subdirectory("${SRC}/lmdb_creator")

# OpenCV
find_package(OpenCV)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set(DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data")
add_definitions(-DDATA_ROOT="${DATA_ROOT}")

foreach(infile ${files})
    get_filename_component(outname ${infile} NAME_WE)
    add_executable(${outname} ${infile} ${SRC}/utils/mnist_utils.hpp ${SRC}/utils/mnist_utils.cpp)
    target_link_libraries(${outname} ${Caffe_LIBRARIES} ${OpenCV_LIBS} lmdb_creator)
endforeach(infile)
