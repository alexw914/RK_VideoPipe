#include <iostream>
#include <string.h>
#include "nodes/vp_rtsp_src_node.h"
#include "nodes/vp_ffmpeg_src_node.h"
#include "nodes/vp_rk_rtsp_src_node.h"
#include "nodes/infer/vp_rk_first_yolo.h"
#include "nodes/infer/vp_rk_second_cls.h"
#include "nodes/vp_fake_des_node.h"

#include "nodes/track/vp_sort_track_node.h"
#include "nodes/broker/vp_json_console_broker_node.h"
#include "nodes/broker/vp_xml_file_broker_node.h"
#include "nodes/osd/vp_osd_node.h"
#include "nodes/vp_screen_des_node.h"
#include "nodes/record/vp_record_node.h"
#include "vp_utils/analysis_board/vp_analysis_board.h"
#include <nlohmann/json.hpp>
#include "yolo.h"

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv) 
{
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_SET_LOG_LEVEL(vp_utils::INFO);
    VP_LOGGER_INIT();

    // create nodes
    auto rtsp_src_0 = std::make_shared<vp_nodes::vp_rk_rtsp_src_node>("rtsp_src_0", 0, "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301");
    auto fake_des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);
    auto rk_yolo_0  = std::make_shared<vp_nodes::vp_rk_first_yolo>("yolo_0", "assets/configs/person.json");
    auto track_0    = std::make_shared<vp_nodes::vp_sort_track_node>("track_0");
    auto osd_0      = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto recorder   = std::make_shared<vp_nodes::vp_record_node>("recorder", "./record", "./record");
    rk_yolo_0->attach_to({rtsp_src_0});
    track_0->attach_to({rk_yolo_0});
    osd_0->attach_to({track_0});
    recorder->attach_to({osd_0});
    fake_des_0->attach_to({recorder});

    /*
    * set hookers for vp_record_node when task compeleted
    */
    // define hooker 
    auto record_hooker = [](int channel, vp_nodes::vp_record_info record_info) {
        auto record_type = record_info.record_type == vp_nodes::vp_record_type::IMAGE ? "image" : "video";

        std::cout << "channel:[" << channel << "] [" <<  record_type << "]" <<  " record task completed! full path: " << record_info.full_record_path << std::endl;
    };
    recorder->set_image_record_complete_hooker(record_hooker);
    recorder->set_video_record_complete_hooker(record_hooker);

    rtsp_src_0->start();
    // for debug purpose
    std::vector<std::shared_ptr<vp_nodes::vp_node>> src_nodes_in_pipe{rtsp_src_0};
    vp_utils::vp_analysis_board board(src_nodes_in_pipe);
    board.display(1, false);  // no block

    /* interact from console */
    /* no except check */
    std::string input;
    std::getline(std::cin, input);
    // input format: `image channel` or `video channel`, like `video 0` means start recording video at channel 0
    auto inputs = vp_utils::string_split(input, ' '); 
    while (inputs[0] != "quit") {
        // no except check
        auto command = inputs[0];
        auto index = std::stoi(inputs[1]);
        auto src_by_channel = std::dynamic_pointer_cast<vp_nodes::vp_src_node>(src_nodes_in_pipe[index]);
        if (command == "video") {
            src_by_channel->record_video_manually(true);   // debug api
            // or
            // src_by_channel->record_video_manually(true, 5);
            // src_by_channel->record_video_manually(false, 20);
        }
        else if (command == "image") {
            src_by_channel->record_image_manually();   // debug api
            // or
            // src_by_channel->record_image_manually(true);
        }
        else {
            std::cout << "invalid command!" << std::endl;
        }
        std::getline(std::cin, input);
        inputs = vp_utils::string_split(input, ' '); 
        if (inputs.size() != 2) {
             std::cout << "invalid input!" << std::endl;
             break;
        }
    }

    rtsp_src_0->detach_recursively();
    // vp_utils::vp_analysis_board board({rtsp_src_0});
    // board.display();

    return 0;
}
