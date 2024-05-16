#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vp_rk_rtsp_src_node.h"
#include "vp_utils/vp_utils.h"

#include "im2d.h"
#include "RgaUtils.h"
#include "im2d_common.h"
#include <chrono>

namespace vp_nodes {
        
    vp_rk_rtsp_src_node::vp_rk_rtsp_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtsp_url, 
                                        float resize_ratio,
                                        int skip_interval): 
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        rtsp_url(rtsp_url), skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        VP_INFO(vp_utils::string_format("[%s] RTSP url: [%s]", node_name.c_str(), to_string().c_str()));
        if (!this->init()){
            VP_INFO(vp_utils::string_format("[%s] Init Demuxer or Decoder failed!", node_name.c_str()));
            exit(0);
        }
        this->initialized();
    }
    
    vp_rk_rtsp_src_node::~vp_rk_rtsp_src_node() {
        deinitialized();
        // m_demux.reset();
        // m_decoder.reset();
    }
    
    // define how to read video from rtsp stream, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_rk_rtsp_src_node::handle_run() {
        while(alive) {
            // check if need work
            gate.knock();
            auto pkt = alloc_av_packet();
            int re = m_demux->read_packet(pkt);
            if (re >= 0 && pkt.get() != nullptr)
            {
                if (pkt->stream_index != m_demux->get_video_stream_index()) {
                    continue;  // 忽略非视频帧
                }
                m_decoder->Decode(pkt->data, pkt->size, 0);
            }
        }
        // send dead flag for dispatch_thread
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    // return stream url
    std::string vp_rk_rtsp_src_node::to_string() {
        return rtsp_url;
    }

    bool vp_rk_rtsp_src_node::init(){
        if (!m_demux) {
            m_demux = FFmpeg::Demuxer::createShare();
        }
        if (!(m_demux->open(rtsp_url))) {
            VP_INFO(vp_utils::string_format("[%s] Open RTSP url from %s failed, please check!", node_name.c_str(), rtsp_url.c_str()));
            return false;
        }
        m_width   = m_demux->get_video_codec_parameters()->width;
        m_height  = m_demux->get_video_codec_parameters()->height;
        m_fps     = m_demux->get_video_stream()->avg_frame_rate.num / m_demux->get_video_stream()->avg_frame_rate.den;
        // m_bitrate = m_demux->get_video_codec_parameters()->bit_rate;

        AVCodecID  codec_id = m_demux->get_video_codec_id();
        int m_type = codec_id == AV_CODEC_ID_H264 ? 264 : 265;

        if (!m_decoder) {
            m_decoder = std::make_shared<MppDecoder>(m_type, m_fps, this, 
            [](void *usrdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data){
                auto start = std::chrono::steady_clock::now();

                vp_rk_rtsp_src_node *ctx = (vp_rk_rtsp_src_node *)usrdata;
                // stream_info_hooker activated if need
                vp_stream_info stream_info {ctx->channel_index, ctx->m_fps, ctx->m_width, ctx->m_height,ctx->to_string()};
                ctx->invoke_stream_info_hooker(ctx->node_name, stream_info);  

                // VP_INFO(vp_utils::string_format("[%s] Frame index: [%d], step: [%d]", ctx->node_name.c_str(), ctx->frame_index, ctx->step));
                if (ctx->step++ != ctx->skip_interval) return;
                ctx->step = 0;
                // cv::Mat frame(height*3/2, width, CV_8UC1, data);

                void *img_buf = nullptr;
                img_buf = malloc(width * height * 3);
                memset(img_buf, 0x00, width * height * 3);
                rga_buffer_t src;
                rga_buffer_t dst;
                
                src = wrapbuffer_virtualaddr((void*)data, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
                dst = wrapbuffer_virtualaddr((void*)img_buf, width, height, RK_FORMAT_RGB_888);
                IM_STATUS STATUS = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);
                if (STATUS != IM_STATUS_SUCCESS){
                    VP_ERROR(vp_utils::string_format("[%s] Convert RGB failed!", ctx->node_name.c_str()));
                    if (img_buf) free(img_buf); 
                    return;
                }
                cv::Mat frame(height, width, CV_8UC3, img_buf);
                
                cv::Mat resize_frame;
                ctx->frame_index++;
                if (ctx->resize_ratio != 1.0) {
                    cv::resize(frame, resize_frame, cv::Size(), ctx->resize_ratio, ctx->resize_ratio);
                }
                else {
                    resize_frame = frame.clone(); // clone!;
                }

                auto out_meta = 
                    std::make_shared<vp_objects::vp_frame_meta>(resize_frame, ctx->frame_index, ctx->channel_index, width, height, ctx->m_fps);
                if (out_meta != nullptr) {
                    ctx->out_queue.push(out_meta);
                    // handled hooker activated if need
                    if (ctx->meta_handled_hooker) {
                        ctx->meta_handled_hooker(ctx->node_name, ctx->out_queue.size(), out_meta);
                    }
                    // important! notify consumer of out_queue in case it is waiting.
                    ctx->out_queue_semaphore.signal();
                    VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", ctx->node_name.c_str(), ctx->out_queue.size()));
                }
                if (img_buf) free(img_buf); 
                auto end = std::chrono::steady_clock::now();
                int dur = std::chrono::duration<double, std::milli>(end - start).count();
                // VP_INFO(vp_utils::string_format("[%s] Frame index: [%d], Callback [%d] ms", ctx->node_name.c_str(), ctx->frame_index, dur));
            });
        }
        return true;        
    }
}
