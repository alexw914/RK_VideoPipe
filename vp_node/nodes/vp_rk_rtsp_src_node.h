#pragma once
#include <string>
#include "base/vp_src_node.h"
#include "Demuxer.h"
#include "mpp_decoder.h"

namespace vp_nodes {
    // rtsp source node, receive video stream via rtsp protocal.
    // example:
    // rtsp://admin:admin12345@192.168.77.110:554/
class vp_rk_rtsp_src_node: public vp_src_node {
    private:
        std::shared_ptr<FFmpeg::Demuxer> m_demux;    // 解封装
        std::shared_ptr<MppDecoder> m_decoder;
        std::string rtsp_url;
        int skip_interval = 0;
        int step = 0;
        // 读取的视频信息
        int  m_width, m_height, m_fps = 0;
        bool init();

    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_rk_rtsp_src_node(std::string node_name, 
                        int channel_index, 
                        std::string rtsp_url, 
                        float resize_ratio = 1.0,
                        int skip_interval = 0);
        ~vp_rk_rtsp_src_node();
        virtual std::string to_string() override;
    };
}