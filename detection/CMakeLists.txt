project(detection)

set(DARK_INC  ${CMAKE_CURRENT_SOURCE_DIR}/inc)
aux_source_directory(src DETECTION_SRCS)
include_directories(${DARK_INC})

link_directories(${OpenCV_LIBS} "${TORCH_LIBRARIES}" ) 

add_library(detection SHARED ${DETECTION_SRCS})
include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(detection EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/inc/Detection_export.h)

target_include_directories(detection PUBLIC ${DARK_INC} )
target_link_libraries(detection PUBLIC ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
