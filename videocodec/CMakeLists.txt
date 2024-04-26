cmake_minimum_required(VERSION 3.4.1)

project(videocodec)

aux_source_directory(./ SRC)

add_library(videocodec STATIC
    ${SRC}
)
target_include_directories(videocodec PUBLIC
    ${MPP_INCLUDES}
)
target_link_libraries(videocodec PUBLIC
    ${COMMON_LIBS}
    ${MPP_LIB}
)