find_package(OpenCV REQUIRED)
#find_package(Torch REQUIRED)

aux_source_directory(. YOLOV3_SRCS)

add_executable(yolov3 ${YOLOV3_SRCS})

target_include_directories(yolov3 PRIVATE "${PROJECT_BINARY_DIR}")
target_link_libraries(yolov3 ${OpenCV_LIBS} detection ${STDCXXFS} )
