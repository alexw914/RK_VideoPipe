# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

RK_VideoPipe 是一个移植到 RK3588 平台（瑞芯微 SoC）的视频分析流水线框架。它提供了硬件加速的视频处理（通过 MPP 进行编解码）、AI 推理（通过 RKNN），以及基于节点架构的计算机视觉应用构建方案。

本项目基于 [VideoPipe](https://github.com/sherlockchou86/VideoPipe) 开源项目，并结合了瑞芯微的硬件加速库。

## 构建命令

```bash
# 标准构建（创建 build/ 目录并编译所有内容）
./build-linux.sh

# 手动构建（用于调试构建问题）
cmake -S . -B build
cmake --build build -j4
cmake --install build

# 运行主程序
build/bin/rk_videopipe
```

**注意**：交叉编译工具链设置为 `aarch64-linux-gnu`。编译器设置见 `build-linux.sh`。

## 高层架构

### 基于节点的流水线框架

核心架构是由处理节点组成的有向无环图（DAG）：

```
源节点 -> 推理节点 -> 跟踪节点 -> 二级推理 -> OSD -> 消息代理 -> 目标节点
```

所有节点继承自 `vp_node` (vp_node/nodes/base/vp_node.h:29) 并实现：
- `handle_frame_meta()`: 逐帧处理
- `handle_frame_meta(batch)`: 批量处理帧（用于推理节点）
- `meta_flow()`: 将元数据传递给下游节点

节点通过 `attach_to({pre_nodes})` 连接，形成流式传输帧和元数据的流水线。

### 节点分类

| 类别 | 位置 | 示例 |
|----------|----------|----------|
| **源节点** | `vp_node/nodes/` | `vp_file_src_node`, `vp_rk_rtsp_src_node`, `vp_ffmpeg_src_node` |
| **推理节点** | `vp_node/nodes/infer/` | `vp_rk_first_yolo`, `vp_rk_second_cls`, `vp_rk_second_rtmpose` |
| **跟踪节点** | `vp_node/nodes/track/` | `vp_sort_track_node`, `vp_byte_track_node` |
| **OSD节点** | `vp_node/nodes/osd/` | `vp_osd_node`, `vp_pose_osd_node` |
| **目标节点** | `vp_node/nodes/base/` | `vp_screen_des_node`, `vp_rtmp_des_node`, `vp_file_des_node` |
| **消息代理节点** | `vp_node/nodes/broker/` | `vp_json_console_broker_node` |

### 推理节点模式

推理节点继承自 `vp_infer_node` (vp_node/nodes/infer/vp_infer_node.h:19) 并实现 4 步流程：
1. **prepare()**: 从帧元数据中提取裁剪区域/图像
2. **preprocess()**: 归一化、缩放、均值/标准差减法（提供默认实现）
3. **infer()**: 运行 RKNN 模型（提供默认实现）
4. **postprocess()**: 解析输出并用结果更新帧元数据

### 初级推理 vs 二级推理

- **PRIMARY** (`vp_primary_infer_node`): 对整帧进行推理（如 YOLO 检测器）
- **SECONDARY** (`vp_secondary_infer_node`): 对初级检测结果裁剪的区域进行推理（如对检测到的物体进行分类）

二级节点通过 vector 参数按 `class_id` 过滤目标（参见 `main.cc:48-49`）。

### 硬件加速组件

| 组件 | 位置 | 用途 |
|-----------|----------|---------|
| **MPP** | `videocodec/mpp_decoder.cpp`, `mpp_encoder.cpp` | 硬件 H.264/H.265 编解码 |
| **RGA** | `3rdparty/librga/` | 2D 图形操作（缩放、格式转换） |
| **RKNN** | `models/rkbase.cpp` | NPU 神经网络推理 |

## 模型配置

模型通过 `assets/configs/` 中的 JSON 文件配置。每个模型配置指定：
- RKNN 模型路径
- 输入/输出维度
- 标签文件
- 预处理参数（mean, std, scale）

`models/` 目录包含以下内容的 C++ 封装：
- `yolo.cpp/h`: YOLO v5-v8 目标检测
- `rtmpose.cpp/h`: 姿态估计
- `classifier.cpp/h`: 图像分类

## 重要文件

- `main.cc`: 演示流水线定义 - 修改此文件可改变流水线结构
- `cmake/common.cmake`: OpenCV 和 FFmpeg 库路径 - 如果库在非标准位置需要更新
- `build-linux.sh`: 构建脚本
- `vp_node/vp_utils/analysis_board/`: 流水线可视化工具

## 开发说明

### 添加新的节点类型

1. 继承适当的基类（`vp_node`, `vp_infer_node`, `vp_src_node`, `vp_des_node`）
2. 实现 `handle_frame_meta()` 进行逐帧处理
3. 对于推理节点：实现 `prepare()` 和 `postprocess()`
4. 在 `vp_node/CMakeLists.txt` 中注册（如需要）

### 帧元数据流

帧元数据（`vp_frame_meta`）携带：
- 原始帧 (cv::Mat)
- 检测到的目标 (`vp_frame_target`)
- 跟踪、姿态、子目标
- 用于录制/流传输的控制元数据

元数据通过 `meta_flow()` 在流水线中传递，并累积来自每个节点的信息。

### 线程模型

每个节点运行两个线程：
- **handle_thread**: 从 `in_queue` 处理传入的元数据
- **dispatch_thread**: 将处理后的元数据推送到 `out_queue` 和下游节点

同步使用信号量（`vp_semaphore`）和互斥锁。

### 跨平台注意事项

这是一个针对 RK3588 的 **aarch64** 构建。代码使用：
- 瑞芯微特定库（MPP, RGA, RKNN runtime）
- 带有 rkmpp 插件的 GStreamer 用于视频处理
- 集成 MPP 的 FFmpeg
- 支持 FreeType 的 OpenCV 4.6+

## 测试

没有统一的测试套件。使用：
- `tests/main_model.cc`: 模型测试
- `tests/main_track.cc`: 跟踪测试
- `python/test_model.py`: Python 推理冒烟测试
- `python/test_count.py`: Python 计数/跟踪测试

验证时，使用 `assets/videos/` 中的示例视频运行完整流水线，并验证 OSD 输出或 RTMP 流。
