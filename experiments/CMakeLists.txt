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

include_directories(${COM})

add_subdirectory(${COM})
add_subdirectory(${MNI})
#TODO: make the other CMakeLists files 
#add_subdirectory(${ALE})
#add_subdirectory(${KIT})
#add_subdirectory(${SCE})
#add_subdirectory(${SF})
