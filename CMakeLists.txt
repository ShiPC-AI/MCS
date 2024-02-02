cmake_minimum_required(VERSION 3.0)
project(MCS)

set(CMAKE_CXX_STANDARD 14)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-std=c++14 -g)

#PCL
find_package(PCL 1.12 REQUIRED)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.12)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# message("PCL_INCLUDE_DIRS=${PCL_INCLUDE_DIRS}")

#OPENCV
find_package(OpenCV 4.2.0)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(Boost REQUIRED COMPONENTS
    thread
)

#BOOST
if(NOT Boost_FOUND)
    message("NOT found Boost")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# # Sub directory
add_subdirectory(base)
add_subdirectory(descriptor)
add_subdirectory(offline_map_process)
add_subdirectory(ground)
add_subdirectory(online_place_recognition)



