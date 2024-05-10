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

        
set(COMMON_LIBS Threads::Threads ${FFmpeg_LIBS} ${OpenCV_LIBS})
