# Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

file(GLOB SRC *pp)
add_library(lmdb_creator SHARED ${SRC})
