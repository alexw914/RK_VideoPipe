# threads
find_package(Threads REQUIRED)

# opencv header：/usr/local/include
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# ffmpeg path
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/include)
file(GLOB_RECURSE FFmpeg_LIBS
        /usr/lib/aarch64-linux-gnu/libav*.so
        /usr/lib/aarch64-linux-gnu/libsw*.so
        /usr/lib/aarch64-linux-gnu/libpostproc.so)


# SDL2
find_package(SDL2)
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS})
    set(SDL2_FOUND_LIBS ${SDL2_LIBRARIES})
else()
    # Try to find SDL2 manually
    include_directories(/usr/include/SDL2)
    set(SDL2_FOUND_LIBS SDL2)
endif()

set(COMMON_LIBS Threads::Threads ${FFmpeg_LIBS} ${OpenCV_LIBS} ${SDL2_FOUND_LIBS})

# MPP (Media Process Platform) for RK3588
# Link using library name so linker picks correct version (.so.0 not .so.1)
set(MPP_LIB rockchip_mpp)
set(MPP_INCLUDES /usr/include)
