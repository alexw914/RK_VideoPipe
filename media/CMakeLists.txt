cmake_minimum_required(VERSION 3.4.1)

project(rk_media)

aux_source_directory(./ SRC)

add_library(rk_media STATIC
    ${SRC}
)

target_include_directories(rk_media PUBLIC
    ${MPP_INCLUDES}
)
target_link_libraries(rk_media PUBLIC
    ${MPP_LIB}
)