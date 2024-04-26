cmake_minimum_required(VERSION 3.4.1)

project(vp_node)

file(GLOB_RECURSE SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/excepts/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/objects/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vp_utils/*.cpp)

add_library(vp_node STATIC
    ${SRC}
)
target_include_directories(vp_node PUBLIC
    ${MPP_INCLUDES}
    ${RGA_INCLUDES}
    ${RKNN_RT_INCLUDES}
)
target_link_libraries(vp_node PUBLIC
    ${COMMON_LIBS}
    stdc++fs
    videocodec
    rknn_models
    bytetrack
)