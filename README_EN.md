# RK_VideoPipe
This repo mainly refered [VideoPipe](https://github.com/sherlockchou86/VideoPipe.git) project.I applied this project in RK3588 platform.
This repo can do some video analysis Task.

### Support Models

It supports Object Detection, Classifier and Pose estimation Task as follows:

| Type| Detection | Classifier | Pose | Track |
|:------|:------------:|:------------:|:------------:|:------------:|
| Models | YOLOv5 to v8 | Any | RTMPose | ByteTrack |


### Introduction

Main folder:
- vp_node (It include Nodes definitions and modules)
- videocodec (video decoder and encoder modules for RockChip)

vp_node文件夹：
- nodes (Defitions of all Node)
  - infer (infer Nodes, you can do combination algorithm such as Object Detection + Object Detection, Object Detection + Classification or Object Detection + KeyPoint Detection)
  - osd (Node for draw images)
  - vp_ffmpeg_src_node.cpp (Nodes for Video codec realized by FFmpeg. You must install rkmpp plugin first.)
  - vp_rk_rtsp_src_node.h  (Nodes for Video codec realized by RockChip mpp.)

### Cases

Just define all Nodes you need, and start it.
```
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv) 
{
    // log
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // 源节点、可使用文件、图片以及rtsp流。需要Gst支持对应插件，若使用FFmpeg需要编译mpp插件，且保证正常工作
    auto src_0 = std::make_shared<vp_nodes::vp_file_src_node>("rtsp_src_0", 0, "assets/videos/person.mp4", 1.0, true, "mppvideodec");
    // auto src_0 = std::make_shared<vp_nodes::vp_rk_rtsp_src_node>("rtsp_src_0", 0, "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301");
    // auto src_0 = std::make_shared<vp_nodes::vp_ffmpeg_src_node>("rtsp_src_0", 0, "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301");

    // 推理初始节点, 一般为目标检测
    auto yolo_0     = std::make_shared<vp_nodes::vp_rk_first_yolo>("rk_yolo_0", "assets/configs/person.json");    
    auto track_0    = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");

    // 第二推理节点，用于处理子目标、例如目标检测后对框内物体再检测或分类
    // auto yolo_sub_0 = std::make_shared<vp_nodes::vp_rk_second_yolo>("rk_yolo_sub_0", "assets/configs/phone.json");
    auto pose_0 = std::make_shared<vp_nodes::vp_rk_second_rtmpose>("rk_rtmpose_0", "assets/configs/rtmpose.json", std::vector<int>{0});
    auto cls_0  = std::make_shared<vp_nodes::vp_rk_second_cls>("rk_cls_0", "assets/configs/stand_sit.json", std::vector<int>{0});

    // 绘制节点    
    auto osd_0      = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");

    // 终止节点、可使用rtmp推流节点、屏幕显示节点或不做任何操作
    // auto des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.3.100:1935/stream");
    // auto des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);
    // auto des_0 = std::make_shared<vp_nodes::vp_file_des_node>("file_des_0", 0, "out");
    auto des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    // 消息节点
    auto msg_broker = std::make_shared<vp_nodes::vp_json_console_broker_node>("broker_0");

    // 节点连接操作
    yolo_0->attach_to({src_0});
    track_0->attach_to({yolo_0});
    cls_0->attach_to({track_0});
    pose_0->attach_to({cls_0});
    osd_0->attach_to({pose_0});
    pose_osd_0->attach_to({osd_0});
    msg_broker->attach_to({pose_osd_0});
    des_0->attach_to({msg_broker});

    src_0->start();
    vp_utils::vp_analysis_board board({src_0});
    board.display();

    return 0;
}
```
![](./assets/sources/sample.png)

### Environmet and How to Build

System
- Ubuntu 22.04 jammy aarch64 / Debain (Tested in Rock5b Armbain and OrangePi5B Debain)

Environment
- C++ 17
- OpenCV >= 4.6 (Need support FreeType)
- GStreamer (support rkmpp pulgin)
- FFmpeg >= 4.3 (need build [FFmpeg](https://github.com/nyanmisaka/ffmpeg-rockchip) support mpp pulgin)

You can configure OpenCV and FFmpeg in cmake/common.cmake. You don't need install other libraries. Build this repo test samples, just run:
```
cd RK_VideoPipe
./build-linux.sh
build/bin/rk_videopipe
```

### Refer

[VideoPipe](https://github.com/sherlockchou86/VideoPipe.git): Definitions and Modules of nodes are borrowed. \
[trt_yolo_video_pipeline](https://github.com/1461521844lijin/trt_yolo_video_pipeline.git) Decoder and Demuxer realized by FFmpeg \
[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) Postprocess and Preprocess of YOLO series. \
[RTMPose-Deploy](https://github.com/HW140701/RTMPose-Deploy) Refer PostPorcess of RTMPose, I replace LetterBox with Affine Transform.