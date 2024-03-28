# RKNN-SERVER

This repo has realized some common models in RockChip SOC, such as 3588. It also supports some hardware acceleration when do letterbox or decode video stream.

### RKNN models

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

https://github.com/MUZLATAN/ffmpeg_rtsp_mpp
https://github.com/airockchip/rknn_model_zoo
https://github.com/ifzhang/ByteTrack
https://github.com/HW140701/RTMPose-Deploy