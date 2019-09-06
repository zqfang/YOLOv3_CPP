
project(yolov3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/detetion/inc)
aux_source_directory(. YOLOV3_SRCS)
add_executable(yolov3 ${YOLOV3_SRCS})

#target_include_directories(yolov3 ${CMAKE_CURRENT_SOURCE_DIR}/detetion/inc)
target_link_libraries(yolov3 detection )
