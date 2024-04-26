# RK_VideoPipe
This repo mainly refered [VideoPipe](https://github.com/sherlockchou86/VideoPipe.git) project.I applied this open source project in RK3588 platform.
This project can do some video stream analysis, such as Object Detection, Image Classification and KeyPoint Detection.

### Support Models

This repo supports Object Detection, Classifier and Pose estimation Task as follows:

| Type| Detection | Classifier | Pose | Track |
|:------|:------------:|:------------:|:------------:|:------------:|
| Models | YOLOv5 to v8 | Any | RTMPose | ByteTrack |


### Introduce

There are four files in tests folder.

| File | Intro |
|:------|:------------:|
|main_ffmpeg/zlmediakit.cc: |Decode video stream use MPP decoder.|
|main_model.cc:| RKNN model samples.|
|main_track.cc:| YOLOv5 + ByteTrack.|

### Build

You can configure OpenCV and FFmpeg in Cmake/common.cmake. You don't need install other libraries. Build this repo test samples, just run:
```
./build-linux.sh
```

### Refer
This repo refer some repositories：

https://github.com/sherlockchou86/VideoPipe.git