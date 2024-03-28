cmake_minimum_required(VERSION 3.4.1)

project(rknn_server)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LIB_ARCH aarch64)

# # skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined ")

# base cmake config
include(${CMAKE_SOURCE_DIR}/cmake/common.cmake)

# include
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/models
    ${CMAKE_SOURCE_DIR}/media
    ${CMAKE_SOURCE_DIR}/bytetrack
)

add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty 3rdparty)
add_subdirectory(${CMAKE_SOURCE_DIR}/bytetrack bytetrack)
add_subdirectory(${CMAKE_SOURCE_DIR}/models rknn_models)
add_subdirectory(${CMAKE_SOURCE_DIR}/media rk_media)

add_executable(rknn_model
    tests/main_model.cc
)
target_include_directories(rknn_model PUBLIC
    ${YAML_INCLUDES}
)
target_link_libraries(rknn_model
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    rk_media
)

add_executable(rknn_track
    tests/main_track.cc
)
target_include_directories(rknn_track PUBLIC
    ${YAML_INCLUDES}
)
target_link_libraries(rknn_track
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    bytetrack
)

add_executable(rknn_ffmpeg
    tests/main_ffmpeg.cc
)
target_include_directories(rknn_ffmpeg PUBLIC
    ${YAML_INCLUDES}
)
target_link_libraries(rknn_ffmpeg
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    rk_media
    bytetrack
)

add_executable(rknn_zlmediakit
    tests/main_zlmediakit.cc
)
target_include_directories(rknn_zlmediakit PUBLIC
    ${MPP_INCLUDES}
    ${YAML_INCLUDES}
    ${ZLMEDIAKIT_INCLUDES}
)
target_link_libraries(rknn_zlmediakit
    ${COMMON_LIBS}
    ${BASE_LIBS}
    rknn_models
    rk_media
    bytetrack
    ${ZLMEDIAKIT_LIB}
)

# # install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_server_${CMAKE_SYSTEM_NAME})
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# install(TARGETS rknn_server DESTINATION .)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS ${ZLMEDIAKIT_LIB} DESTINATION lib)
install(PROGRAMS ${YAML_LIB} DESTINATION lib)
