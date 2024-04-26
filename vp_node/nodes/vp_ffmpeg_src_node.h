#pragma once
#include <string>
#include "base/vp_src_node.h"
#include "Demuxer.h"
#include "Decoder.h"
#include "Scaler.h"

// some device can use it if rkmpp can be used correctly. Sometimes it will cause RGA error!

namespace vp_nodes {
    // rtsp source node, receive video stream via rtsp protocal.
    // example:
    // rtsp://admin:admin12345@192.168.77.110:554/
class vp_ffmpeg_src_node: public vp_src_node {
    private:
        std::shared_ptr<FFmpeg::Scaler>  m_scaler;   // 视频缩放、格式转换
        std::shared_ptr<FFmpeg::Demuxer> m_demux;    // 解封装
        std::shared_ptr<FFmpeg::Decoder> m_decoder;  // 解码
        std::string rtsp_url;
        int skip_interval = 0;
        int step = 0;
        // 读取的视频信息
        int  m_width, m_height, m_fps = 0;
        bool m_cycle = true;
        bool init();

    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_ffmpeg_src_node(std::string node_name, 
                        int channel_index, 
                        std::string rtsp_url, 
                        float resize_ratio = 1.0,
                        int skip_interval = 0);
        ~vp_ffmpeg_src_node();
        virtual std::string to_string() override;
    };
}