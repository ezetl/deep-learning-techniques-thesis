set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++11")
add_definitions ("-Wall")

set(SRC ".")
set(files "${SRC}/mnist/preprocess_mnist_siamese.cpp" "${SRC}/mnist/preprocess_mnist_standar.cpp" )

include_directories("${SRC}/lmdb_creator")
add_subdirectory("${SRC}/lmdb_creator")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set(DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../data")
add_definitions(-DDATA_ROOT="${DATA_ROOT}")

foreach(infile ${files})
    get_filename_component(outname ${infile} NAME_WE)
    add_executable(${outname} ${infile} ${SRC}/mnist/mnist_utils.hpp ${SRC}/mnist/mnist_utils.cpp)
    target_link_libraries(${outname} ${Caffe_LIBRARIES} ${OpenCV_LIBS} lmdb_creator)
endforeach(infile)

add_executable(preprocess_kitti_siamese "${SRC}/kitti/preprocess_kitti_siamese.cpp")
target_link_libraries(preprocess_kitti_siamese ${Caffe_LIBRARIES} ${OpenCV_LIBS} lmdb_creator)

# cp sun387 scripts
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sun397/create_SUN_splits.py" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/create_SUN_splits" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sun397/preprocess_SUN.py" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/preprocess_SUN" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sun397/create_SUN_lmdbs.sh" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/create_SUN_lmdbs" @ONLY)

# cp utils folder
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/utils" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
