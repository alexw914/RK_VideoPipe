#include <iostream>
#include <filesystem>

#include "nodes/vp_ffmpeg_src_node.h"
#include "nodes/vp_file_src_node.h"
#include "nodes/vp_rk_rtsp_src_node.h"

#include "nodes/infer/vp_rk_first_yolo.h"
#include "nodes/infer/vp_rk_second_yolo.h"
#include "nodes/infer/vp_rk_second_cls.h"
#include "nodes/infer/vp_rk_second_rtmpose.h"

#include "nodes/vp_fake_des_node.h"
#include "nodes/vp_rtmp_des_node.h"
#include "nodes/vp_file_des_node.h"
#include "nodes/vp_screen_des_node.h"

#include "nodes/track/vp_sort_track_node.h"
#include "nodes/track/vp_byte_track_node.h"
#include "nodes/broker/vp_json_console_broker_node.h"

#include "nodes/osd/vp_osd_node.h"
#include "nodes/osd/vp_pose_osd_node.h"
#include "vp_utils/analysis_board/vp_analysis_board.h"

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv) 
{
    //std::filesystem::current_path("/root/.vs/RK_VideoPipe/41b606b4-6586-4527-af89-a26a0ab1539d/src");
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto src_0 = std::make_shared<vp_nodes::vp_file_src_node>("rtsp_src_0", 0, "assets/videos/person.mp4", 1.0, true, "mppvideodec");
    // auto src_0 = std::make_shared<vp_nodes::vp_rk_rtsp_src_node>("rtsp_src_0", 0, "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301");
    // auto src_0 = std::make_shared<vp_nodes::vp_ffmpeg_src_node>("rtsp_src_0", 0, "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301");

    auto yolo_0  = std::make_shared<vp_nodes::vp_rk_first_yolo>("rk_yolo_0", "assets/configs/person.json");    
    //auto track_0 = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");
    auto track_0 = std::make_shared<vp_nodes::vp_byte_track_node>("track_0");

    //auto yolo_sub_0 = std::make_shared<vp_nodes::vp_rk_second_yolo>("rk_yolo_sub_0", "assets/configs/phone.json");
    auto pose_0 = std::make_shared<vp_nodes::vp_rk_second_rtmpose>("rk_rtmpose_0", "assets/configs/rtmpose.json", std::vector<int>{0});
    auto cls_0  = std::make_shared<vp_nodes::vp_rk_second_cls>("rk_cls_0", "assets/configs/stand_sit.json", std::vector<int>{0});
    
    auto osd_0      = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");

    //auto des_0 = std::make_shared<vp_nodes::vp_rtmp_des_node>("rtmp_des_0", 0, "rtmp://192.168.3.100:1935/stream");
    //auto des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);
    //auto des_0 = std::make_shared<vp_nodes::vp_file_des_node>("file_des_0", 0, "out");
    auto des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    auto msg_broker = std::make_shared<vp_nodes::vp_json_console_broker_node>("broker_0");

    yolo_0->attach_to({src_0});
    track_0->attach_to({yolo_0});
    cls_0->attach_to({track_0});
    pose_0->attach_to({cls_0});
    osd_0->attach_to({pose_0});
    pose_osd_0->attach_to({osd_0});
    msg_broker->attach_to({pose_osd_0});
    des_0->attach_to({msg_broker});

    src_0->start();

    //getchar();
    //src_0->detach_recursively();
    vp_utils::vp_analysis_board board({src_0});
    board.display();

    return 0;
}
