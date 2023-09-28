cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/include/runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/include/runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588 or rv110x")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
  endif()
endif()
find_package(OpenCV REQUIRED)

#ffmpeg
set(FFmpeg_LIBS /usr/lib/aarch64-linux-gnu)
set(ffmpeg_headers_DIR /usr/include/aarch64-linux-gnu)

add_library( avcodec SHARED IMPORTED)
add_library( avfilter SHARED IMPORTED )
add_library( swresample SHARED IMPORTED )
add_library( swscale SHARED IMPORTED )
add_library( avformat SHARED IMPORTED )
add_library( avutil SHARED IMPORTED )


#指定所添加依赖库的导入路径
set_target_properties( avcodec PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libavcodec.so )
set_target_properties( avfilter PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libavfilter.so )
set_target_properties( swresample PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libswresample.so )
set_target_properties( swscale PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libswscale.so )
set_target_properties( avformat PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libavformat.so )
set_target_properties( avutil PROPERTIES IMPORTED_LOCATION ${FFmpeg_LIBS}/libavutil.so )


# 添加头文件路径到编译器的头文件搜索路径下，多个路径以空格分隔
include_directories( /usr/include/aarch64-linux-gnu )
link_directories(${FFmpeg_LIBS} )
# find_package(FFmpeg REQUIRED)


#rga
if(TARGET_SOC STREQUAL "rk356x")
  set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK356X)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588")
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RGA_LIB ${RGA_PATH}/lib/Android/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)
endif()
include_directories( ${RGA_PATH}/include)

# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/3rdparty/mpp)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librockchip_mpp.so)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/libmpp.so)
endif()
include_directories(${MPP_PATH}/include)

# zlmediakit
set(ZLMEDIAKIT_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/zlmediakit)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
include_directories(${ZLMEDIAKIT_PATH}/include)
set(ZLMEDIAKIT_LIBS ${ZLMEDIAKIT_PATH}/${LIB_ARCH}/libmk_api.so)
endif()

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_stream
  src/main_stream.cc
  yolo/postprocessor.cpp
  yolo/rkyolov5.cpp
  yolo/resize_function.cc
  utils/mpp_decoder.cpp
)

target_link_libraries(rknn_yolov5_stream
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  ${MPP_LIBS}
  avcodec avformat avutil swresample swscale swscale avfilter
)

add_executable(rknn_yolov5_main
  src/main.cc
  yolo/postprocessor.cpp
  yolo/rkyolov5.cpp
  yolo/resize_function.cc
)

target_link_libraries(rknn_yolov5_main
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

add_executable(rknn_yolov5_player
  src/main_player.cc
  yolo/postprocessor.cpp
  yolo/rkyolov5.cpp
  yolo/resize_function.cc
  player/player.cpp
  utils/mpp_encoder.cpp
  utils/mpp_decoder.cpp
  utils/drawing.cpp
)

target_link_libraries(rknn_yolov5_player
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  ${MPP_LIBS}
  ${ZLMEDIAKIT_LIBS}
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

install(TARGETS rknn_yolov5_main DESTINATION ./)
install(TARGETS rknn_yolov5_stream DESTINATION ./)
install(TARGETS rknn_yolov5_player DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS ${ZLMEDIAKIT_LIBS} DESTINATION lib)

install(DIRECTORY model DESTINATION ./)