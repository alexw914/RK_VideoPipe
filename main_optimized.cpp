#include <iostream>
#include <filesystem>
#include <cstring>
#include <csignal>
#include <atomic>
#include <thread>

// 全局信号接收标志
// 用于指示是否接收到退出信号（SIGINT/SIGTERM）
static std::atomic<bool> g_signal_received(false);

// 信号处理函数
// 处理 SIGINT (Ctrl+C) 和 SIGTERM 信号，设置全局退出标志
//
// @param signal 接收到的信号编号
void signal_handler(int signal) {
    g_signal_received = true;
}

// ============================================================================
// 头文件包含
// ============================================================================

// 源节点
#include "nodes/vp_mpp_file_src_node.h"        // MPP 硬件解码器源节点
#include "nodes/vp_sdl2_des_node.h"             // SDL2 硬件渲染目标节点
#include "nodes/vp_file_src_node.h"             // 文件源节点（GStreamer）
#include "nodes/vp_ffmpeg_src_node.h"           // FFmpeg 源节点
#include "nodes/vp_rk_rtsp_src_node.h"          // RTSP 源节点

// 推理节点
#include "nodes/infer/vp_rk_first_yolo.h"       // YOLO 目标检测（初级推理）
#include "nodes/infer/vp_rk_second_yolo.h"      // YOLO 目标检测（二级推理）
#include "nodes/infer/vp_rk_second_cls.h"       // 图像分类（二级推理）
#include "nodes/infer/vp_rk_second_rtmpose.h"   // RTMPose 姿态估计（二级推理）

// 目标节点
#include "nodes/vp_fake_des_node.h"             // 虚拟目标节点（空操作）
#include "nodes/vp_rtmp_des_node.h"             // RTMP 推流目标节点
#include "nodes/vp_file_des_node.h"             // 文件录制目标节点
#include "nodes/vp_screen_des_node.h"           // 屏幕显示目标节点（OpenCV）

// 跟踪节点
#include "nodes/track/vp_sort_track_node.h"    // SORT 跟踪算法
#include "nodes/track/vp_byte_track_node.h"    // ByteTrack 跟踪算法

// 消息代理节点
#include "nodes/broker/vp_json_console_broker_node.h"  // JSON 控制台输出

// OSD 节点
#include "nodes/osd/vp_osd_node.h"              // OSD 绘制节点
#include "nodes/osd/vp_pose_osd_node.h"        // 姿态 OSD 绘制节点

// 分析板（流水线可视化）
#include "vp_utils/analysis_board/vp_analysis_board.h"

// ============================================================================
// 主函数
// ============================================================================

// 打印程序使用说明
//
// @param prog_name 程序名称
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -f <file>       输入视频文件路径 (默认: /mnt/nfs/datasets/video/uav.mp4)\n"
              << "  -m <mode>       运行模式:\n"
              << "                    0 = 完整 AI 流水线 (默认)\n"
              << "                    1 = 高性能解码+显示 (无 AI 处理)\n"
              << "  -v               启用 SDL2 垂直同步 (降低 FPS)\n"
              << "  -D <driver>      SDL2 视频驱动 (如 x11, wayland, kmsdrm)\n"
              << "  -R <driver>      SDL2 渲染驱动 (如 opengl, opengles2)\n"
              << "  -fps             显示 FPS 覆盖层\n"
              << "  -h               显示此帮助信息\n";
    std::cout << "\n性能对比:\n"
              << "  模式 0: 完整 AI 流水线 (YOLO + 跟踪 + 姿态估计)\n"
              << "  模式 1: 纯硬件解码 + 显示 (最大 FPS，可达 ~170 FPS)\n";
}

// 主函数
//
// @param argc 参数个数
// @param argv 参数列表
// @return 程序退出码
int main(int argc, char** argv)
{
    // ----------------------------------------------------------------------
    // 默认参数配置
    // ----------------------------------------------------------------------
    const char* input_file = "/mnt/nfs/datasets/video/uav.mp4";  // 输入视频文件
    int mode = 0;              // 运行模式: 0=完整AI流水线, 1=高性能解码+显示
    bool enable_vsync = false; // 是否启用垂直同步
    bool show_fps = true;      // 是否显示 FPS 覆盖层
    const char* sdl_video_driver = "";    // SDL2 视频驱动
    const char* sdl_render_driver = "";  // SDL2 渲染驱动

    // ----------------------------------------------------------------------
    // 解析命令行参数
    // ----------------------------------------------------------------------
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            enable_vsync = true;
        } else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) {
            sdl_video_driver = argv[++i];
        } else if (strcmp(argv[i], "-R") == 0 && i + 1 < argc) {
            sdl_render_driver = argv[++i];
        } else if (strcmp(argv[i], "-fps") == 0) {
            show_fps = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "未知选项: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // ----------------------------------------------------------------------
    // 初始化日志系统
    // ----------------------------------------------------------------------
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);  // 不显示代码位置
    VP_SET_LOG_INCLUDE_THREAD_ID(false);      // 不显示线程 ID
    VP_SET_LOG_LEVEL(vp_utils::INFO);         // 日志级别: INFO
    VP_LOGGER_INIT();                          // 初始化日志器

    // ----------------------------------------------------------------------
    // 打印程序配置信息
    // ----------------------------------------------------------------------
    std::cout << "==========================================\n";
    std::cout << "RK_VideoPipe - 优化版 MPP+SDL2\n";
    std::cout << "==========================================\n";
    std::cout << "输入文件: " << input_file << "\n";
    std::cout << "运行模式: " << (mode == 0 ? "完整 AI 流水线" : "高性能解码+显示") << "\n";
    std::cout << "垂直同步: " << (enable_vsync ? "启用" : "禁用 (最大吞吐量)") << "\n";
    std::cout << "==========================================\n\n";

    // ----------------------------------------------------------------------
    // 模式 0: 完整 AI 流水线
    // ----------------------------------------------------------------------
    //
    // 此模式包含 YOLO 目标检测、跟踪、姿态估计、分类、OSD 和消息代理。
    // 预期 FPS: 15-30 (取决于模型复杂度)
    //
    // 流水线结构:
    //   文件源 -> YOLO检测 -> ByteTrack -> RTMPose -> 分类 -> OSD -> 姿态OSD -> 消息代理 -> SDL2显示
    // ----------------------------------------------------------------------
    if (mode == 0) {
        std::cout << "创建完整 AI 流水线...\n";

        // 创建源节点 (使用 MPP 硬件解码器)
        auto src_0 = std::make_shared<vp_nodes::vp_file_src_node>(
            "file_src_0", 0, input_file, 1.0, true, "mppvideodec");

        // 创建推理节点
        auto yolo_0 = std::make_shared<vp_nodes::vp_rk_first_yolo>("rk_yolo_0", "assets/configs/person.json");
        auto track_0 = std::make_shared<vp_nodes::vp_byte_track_node>("track_0");
        auto pose_0 = std::make_shared<vp_nodes::vp_rk_second_rtmpose>(
            "rk_rtmpose_0", "assets/configs/rtmpose.json", std::vector<int>{0});
        auto cls_0 = std::make_shared<vp_nodes::vp_rk_second_cls>(
            "rk_cls_0", "assets/configs/stand_sit.json", std::vector<int>{0});

        // 创建 OSD 节点
        auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
        auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");

        // 创建消息代理节点
        auto msg_broker = std::make_shared<vp_nodes::vp_json_console_broker_node>("broker_0");

        // 创建 SDL2 显示目标节点 (使用硬件加速渲染)
        auto des_0 = std::make_shared<vp_nodes::vp_sdl2_des_node>(
            "sdl2_des_0", 0, true, show_fps, enable_vsync,
            sdl_video_driver, sdl_render_driver);

        // ----------------------------------------------------------------------
        // 连接流水线节点
        // ----------------------------------------------------------------------
        yolo_0->attach_to({ src_0 });           // 源 -> YOLO检测
        track_0->attach_to({ yolo_0 });         // YOLO -> 跟踪
        cls_0->attach_to({ track_0 });          // 跟踪 -> 分类
        pose_0->attach_to({ cls_0 });           // 分类 -> 姿态估计
        osd_0->attach_to({ pose_0 });           // 姿态 -> OSD
        pose_osd_0->attach_to({ osd_0 });       // OSD -> 姿态OSD
        msg_broker->attach_to({ pose_osd_0 });  // 姿态OSD -> 消息代理
        des_0->attach_to({ msg_broker });       // 消息代理 -> 显示

        // ----------------------------------------------------------------------
        // 启动流水线
        // ----------------------------------------------------------------------
        src_0->start();

        std::cout << "\n完整 AI 流水线运行中...\n";
        std::cout << "流水线: 文件源 -> YOLO -> ByteTrack -> RTMPose -> 分类 -> OSD -> 显示\n";
        std::cout << "按 ESC 或关闭窗口退出\n";
        std::cout << "按 Ctrl+C 可随时退出\n\n";

        // 注册信号处理器 (用于 Ctrl+C)
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // 启动分析板 (非阻塞模式)
        // 分析板会显示流水线结构的可视化窗口
        vp_utils::vp_analysis_board board({ src_0 });
        board.display(1, false);  // 非阻塞显示

        // ----------------------------------------------------------------------
        // 等待退出信号
        // ----------------------------------------------------------------------
        while (!g_signal_received && src_0->is_alive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\n流水线结束。\n";

    // ----------------------------------------------------------------------
    // 模式 1: 高性能解码+显示
    // ----------------------------------------------------------------------
    //
    // 此模式跳过所有 AI 处理，以实现最大 FPS。
    // 直接路径: MPP 硬件解码 -> SDL2 显示
    // 预期 FPS: 150-170 (类似 mp4_hw_dec_sdl2 示例)
    //
    // 流水线结构:
    //   MPP文件源 -> SDL2显示
    // ----------------------------------------------------------------------
    } else if (mode == 1) {
        std::cout << "创建高性能解码+显示流水线...\n";

        // 创建 MPP 硬件解码器源节点
        // 使用 FFmpeg 进行解封装 + MPP 进行硬件解码
        // 直接输出 NV12 格式以实现零拷贝渲染
        auto src_0 = std::make_shared<vp_nodes::vp_mpp_file_src_node>(
            "mpp_src_0", 0, input_file, true);

        // 创建 SDL2 硬件渲染目标节点
        // 使用 SDL2 直接渲染 NV12 格式帧
        auto des_0 = std::make_shared<vp_nodes::vp_sdl2_des_node>(
            "sdl2_des_0", 0, false, show_fps, enable_vsync,
            sdl_video_driver, sdl_render_driver);

        // ----------------------------------------------------------------------
        // 直接连接: 源节点 -> 目标节点 (无中间节点)
        // ----------------------------------------------------------------------
        des_0->attach_to({ src_0 });

        // ----------------------------------------------------------------------
        // 启动流水线
        // ----------------------------------------------------------------------
        src_0->start();

        std::cout << "\n高性能流水线运行中...\n";
        std::cout << "流水线: MPP 文件源 -> SDL2 显示\n";
        std::cout << "目标 FPS: ~170 FPS (类似 mp4_hw_dec_sdl2)\n";
        std::cout << "\n功能特性:\n";
        std::cout << "  - 数据流可视化窗口 (左侧) 显示流水线结构\n";
        std::cout << "  - SDL2 视频显示 (右侧) 渲染解码后的帧\n";
        std::cout << "  - FPS 覆盖层显示在视频上\n";
        std::cout << "\n退出方式:\n";
        std::cout << "  - 在数据流窗口或 SDL 窗口中按 ESC 键\n";
        std::cout << "  - 按 Ctrl+C 随时退出\n";
        std::cout << "  - 视频播放完成后程序自动退出\n\n";

        // 注册信号处理器 (用于 Ctrl+C)
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // 启动分析板 (非阻塞模式)
        // 这允许我们定期检查退出条件
        vp_utils::vp_analysis_board board({ src_0 });
        board.display(1, false);  // 非阻塞显示

        // ----------------------------------------------------------------------
        // 等待任意退出条件:
        // 1. 接收到信号 (Ctrl+C)
        // 2. 视频源播放完成 (src_0->finished)
        // 3. 在 SDL 窗口中按了 ESC (des_0->is_alive() 变为 false)
        // 4. 在分析板窗口中按了 ESC (由 board 内部处理)
        // ----------------------------------------------------------------------
        while (!g_signal_received && !src_0->finished && des_0->is_alive() && src_0->is_alive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // ----------------------------------------------------------------------
        // 打印退出原因
        // ----------------------------------------------------------------------
        if (g_signal_received) {
            std::cout << "\n接收到退出信号，正在退出...\n";
        } else if (src_0->finished) {
            std::cout << "\n视频源播放完成，正在退出...\n";
        } else if (!des_0->is_alive()) {
            std::cout << "\nSDL 窗口已关闭，正在退出...\n";
        } else {
            std::cout << "\n流水线结束。\n";
        }
    }

    return 0;
}
