cmake_minimum_required(VERSION 3.4.1)

project(bytetrack)

aux_source_directory(./ SRC)

add_library(bytetrack STATIC
    ${SRC}
)
target_include_directories(bytetrack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen3
)
target_link_libraries(bytetrack
    ${OpenCV_LIBS}
)



