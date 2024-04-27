# RK_VideoPipe
本项目主要参考[VideoPipe](https://github.com/sherlockchou86/VideoPipe.git)开源项目, 将其移植到RK3588平台，搭配硬件编解码，可用来构建视频分析应用。

### 模型支持

项目支持瑞芯微RKNN平台目标检测、图像分类以及关键点检测等视频分析应用，支持模型如下表。在models文件夹下定义了各类模型的实现方法。

| 类型 | 目标检测 | 分类 | 关键点 | 目标追踪 |
|:------|:------------:|:------------:|:------------:|:------------:|
| 模型 | YOLOv5至v8 | 任意 | RTMPose | ByteTrack |

### 功能介绍

主要文件夹目录：
- vp_node (定义了支持节点推理的各个模块、参考自VideoPipe项目, 重新定义了拉流节点和大部分推理节点)
- videocodec (RK平台视频流处理，硬件码来自官网案例, 并参考了[trt_yolo_video_pipeline](https://github.com/1461521844lijin/trt_yolo_video_pipeline.git)项目的FFmpeg编解码实现)
  
vp_node文件夹：
- nodes (各个节点定义)
  - infer (定义了推理节点，根据节点处理顺序可实现目标检测+分类、目标检测+关键点、目标检测+目标检测三类组合任务)
  - osd (绘制节点、可绘制目标检测、关键点)
  - vp_ffmpeg_src_node.cpp (使用FFmpeg完成拉流或者文件读写节点、隔帧检测时可控制yuv转换rgb过程，降低cpu损耗, 这个需要保证hevc_rkmpp以及h264_rkmpp等插件正常使用，部分机器安装了插件有时也会产生RGA大于4G错误，不知道如何解决)
  - vp_rk_rtsp_src_node.h  (视频流读取节点，使用mpp实现硬件编解码过程，也可控制yuv至rgb转换过程, 不需要使用插件)

### 实现案例

在程序中定义整个节点流向即可，案例如下：
```
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv) 
{
    // 定义log
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
### 项目构建

平台
- Ubuntu 22.04 jammy aarch64 / Debain (已测试香橙派5B平台ubuntu系统和Rock5B平台Armbain系统)

环境
- C++ 17
- OpenCV >= 4.6 (需支持FreeType, 否则需要改写部分OSD节点)
- GStreamer (官网推荐完整安装，需额外支持rkmpp插件)
- FFmpeg >= 4.3 (需要编译mpp插件或使用推荐源)

需要校对cmake目录下的common.cmake文件中定义了FFmpeg与OpenCV位置, 如果不符合则需要修改。构建项目后执行build/bin下可执行文件即可运行案例
```
cd RK_VideoPipe
./build-linux.sh
build/bin/rk_videopipe
```

### 参考项目

[VideoPipe](https://github.com/sherlockchou86/VideoPipe.git): 主要参考项目，大部分节点定义和实现均由该仓库提供 \
[trt_yolo_video_pipeline](https://github.com/1461521844lijin/trt_yolo_video_pipeline.git) 参考了FFmpeg的编解码的实现 \
[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 参考了YOLO系列和分类模型实现，完成了C++类封装 \
[RTMPose-Deploy](https://github.com/HW140701/RTMPose-Deploy) 参考了RTMPose后处理方案，放弃了仿射变换实现，转用LetterBox实现。
