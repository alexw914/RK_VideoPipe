#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vp_ffmpeg_src_node.h"
#include "vp_utils/vp_utils.h"

#include "im2d.h"
#include "RgaUtils.h"
#include "im2d_common.h"
#include <chrono>

namespace vp_nodes {
        
    vp_ffmpeg_src_node::vp_ffmpeg_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtsp_url, 
                                        float resize_ratio,
                                        int skip_interval): 
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        rtsp_url(rtsp_url), skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        VP_INFO(vp_utils::string_format("[%s]", node_name.c_str()));
        if (!this->init()){
            VP_INFO(vp_utils::string_format("[%s] Init Demuxer or Decoder failed!", node_name.c_str()));
            exit(0);
        }
        this->initialized();
    }
    
    vp_ffmpeg_src_node::~vp_ffmpeg_src_node() {
        deinitialized();
        // m_demux.reset();
        // m_decoder.reset();
        // m_scaler.reset();
    }
    
    // define how to read video from rtsp stream, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_ffmpeg_src_node::handle_run() {
        int skip = 0;
        while (alive)
        {
            // check if need work
            gate.knock();
            // stream_info_hooker activated if need
            vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
            invoke_stream_info_hooker(node_name, stream_info);

            auto pkt = alloc_av_packet();
            int re = m_demux->read_packet(pkt);
            if (re == EXIT_SUCCESS && pkt.get() != nullptr) {
                if (pkt->stream_index != m_demux->get_video_stream_index()) {
                    continue;  // 忽略非视频帧
                }
                m_decoder->send(pkt);
                auto frame = alloc_av_frame();
                if (!m_decoder->receive(frame)) {
                    continue;  // 编码器前几帧的缓存可能接收不到
                }
                // need skip
                if (skip < skip_interval) {
                    skip++;
                    continue;
                }
                skip = 0;

                cv::Mat image(frame->height, frame->width, CV_8UC3);
                if (!m_scaler->scale<av_frame, cv::Mat>(frame, image)) {
                    VP_ERROR(vp_utils::string_format("[%s] Run Scaler Failed!", node_name.c_str()));
                    continue;
                }
                cv::Mat resize_frame;
                if (this->resize_ratio != 1.0f) {                 
                    cv::resize(image, resize_frame, cv::Size(), resize_ratio, resize_ratio);
                }
                else {
                    resize_frame = image.clone(); // clone!;
                }

                // set true size because resize
                m_width = resize_frame.cols;
                m_height = resize_frame.rows;
                this->frame_index++;
                // create frame meta
                auto out_meta = 
                    std::make_shared<vp_objects::vp_frame_meta>(resize_frame, this->frame_index, this->channel_index, m_width, m_height, m_fps);

                if (out_meta != nullptr) {
                    this->out_queue.push(out_meta);
                    
                    // handled hooker activated if need
                    if (this->meta_handled_hooker) {
                        meta_handled_hooker(node_name, out_queue.size(), out_meta);
                    }

                    // important! notify consumer of out_queue in case it is waiting.
                    this->out_queue_semaphore.signal();
                    VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
                } 
            } else if (re == AVERROR_EOF) {
                VP_INFO(vp_utils::string_format("[%s] Read RTSP url from %s AVERROR_EOF!", node_name.c_str(), rtsp_url.c_str()));
                if (m_cycle) {
                    m_demux->seek(0);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    continue;
                }
                break;
            } else {
                VP_ERROR(vp_utils::string_format("[%s] Read RTSP url from %s ERROR!", node_name.c_str(), rtsp_url.c_str()));
                break;
            }
        }
        // send dead flag for dispatch_thread
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    // return stream url
    std::string vp_ffmpeg_src_node::to_string() {
        return rtsp_url;
    }

    bool vp_ffmpeg_src_node::init(){

        if (!m_demux) {
            m_demux = FFmpeg::Demuxer::createShare();
        }
        if (!(m_demux->open(rtsp_url))) {
            VP_INFO(vp_utils::string_format("[%s] Open RTSP url from %s failed, please check!", node_name.c_str(), rtsp_url.c_str()));
            return false;
        }
        if (!m_scaler) {
            m_scaler = FFmpeg::Scaler::createShare(
                m_demux->get_video_codec_parameters()->width,
                m_demux->get_video_codec_parameters()->height,
                (AVPixelFormat)m_demux->get_video_codec_parameters()->format,
                m_demux->get_video_codec_parameters()->width,
                m_demux->get_video_codec_parameters()->height, AV_PIX_FMT_BGR24);
        }
        if (!m_decoder) {
            m_decoder = FFmpeg::Decoder::createShare(m_demux);
        }
        if (!(m_decoder->open(false))) {
            return false;
        }
        m_width   = m_demux->get_video_codec_parameters()->width;
        m_height  = m_demux->get_video_codec_parameters()->height;
        m_fps     = m_demux->get_video_stream()->avg_frame_rate.num / m_demux->get_video_stream()->avg_frame_rate.den;
        
        return true;        
    }
}