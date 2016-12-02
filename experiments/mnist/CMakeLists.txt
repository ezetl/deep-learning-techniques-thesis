set(FILES "preprocess_mnist_siamese.cpp" "preprocess_mnist_standar.cpp")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++11")

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

foreach(infile ${FILES})
    get_filename_component(outname ${infile} NAME_WE)
    add_executable(${outname} ${infile} mnist_utils.hpp mnist_utils.cpp)
    target_link_libraries(${outname} ${Caffe_LIBRARIES} ${OpenCV_LIBS} lmdb_creator)
endforeach(infile)
