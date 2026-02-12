cmake_minimum_required(VERSION 3.4.1)

project(rk_videopipe)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LIB_ARCH aarch64)

# # skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined ")

# # install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/build/lib)

# base cmake config
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty 3rdparty)
# include
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/models
    ${CMAKE_SOURCE_DIR}/media
    ${CMAKE_SOURCE_DIR}/bytetrack
    ${CMAKE_SOURCE_DIR}/videocodec
    ${CMAKE_SOURCE_DIR}/vp_node
    ${BASE_INCLUDES}
)

add_subdirectory(${CMAKE_SOURCE_DIR}/models rknn_models)
add_subdirectory(${CMAKE_SOURCE_DIR}/bytetrack bytetrack)
add_subdirectory(${CMAKE_SOURCE_DIR}/videocodec videocodec)
add_subdirectory(${CMAKE_SOURCE_DIR}/vp_node vp_node)

add_executable(rk_videopipe
    main.cc
)
target_link_libraries(rk_videopipe
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    videocodec
    vp_node
)

# Optimized version with MPP+SDL2 support
add_executable(rk_videopipe_opt
    main_optimized.cpp
)
target_link_libraries(rk_videopipe_opt
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    videocodec
    vp_node
)

