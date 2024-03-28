cmake_minimum_required(VERSION 3.4.1)

project(rknn_models)

aux_source_directory(./ SRC)

add_library(rknn_models STATIC
    ${SRC}
)
target_include_directories(rknn_models PUBLIC
    ${RKNN_RT_INCLUDES}
    ${RGA_INCLUDES}
)
target_link_libraries(rknn_models
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    ${RGA_LIB}
)



