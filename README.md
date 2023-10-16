# rknn_stream

## Soc: RK3588
## detect the real-time stream video using yolov5 (include rtsp decode and RKNN infer)

Three main function
  ##### --main.cc        opencv gstreamer backend
  ##### --main_stream.cc ffmpeg --version 4.3
  ##### --main_player.cc Zlmediakit

Refer repositories：
  ##### https://github.com/MUZLATAN/ffmpeg_rtsp_mpp
  ##### https://github.com/rockchip-linux/rknpu2
  ##### https://github.com/airockchip/rknn_model_zoo
