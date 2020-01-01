cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(YOLOV3-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "/Users/zqfang/Github/YOLOv3_CPP/libtorch/share/cmake/Torch")
#set(OpenCV_DIR D:/workspace/opencv410/build)
#set(CMAKE_PREFIX_PATH "D:/fzq/YOLOv3_CPP/libtorch120debug")

# set(OUTPUT_DIR "result")
# set(TARGETS_DIR_NAME "targets")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# For profiler on Ubuntu
if (CMAKE_BUILD_TYPE STREQUAL Debug AND NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()

# Output directory structure
# set(OUTPUT_DIR "result")
# set(TARGETS_DIR_NAME "targets")

# configure_file(
#         "${PROJECT_SOURCE_DIR}/config.h.in"
#         "${PROJECT_BINARY_DIR}/config.h"
# )

# GCC need to link against stdc++fs
if(MSVC)
set(STDCXXFS "")
else()
set(STDCXXFS "stdc++fs")
endif()

# .exe and .dll
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(yolov3)
add_subdirectory(detection)


