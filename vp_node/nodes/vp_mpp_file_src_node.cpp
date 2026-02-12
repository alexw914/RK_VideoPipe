#include <iostream>
#include <chrono>
#include <thread>

#include "vp_utils/logger/vp_logger.h"
#include "vp_utils/vp_utils.h"
#include "vp_mpp_file_src_node.h"

namespace vp_nodes {

static const size_t kInputChunkSize = 4096;

// Map FFmpeg codec to MPP coding type
static int map_dec_codec(AVCodecID cid, MppCodingType* coding, const char** bsf_name) {
    switch (cid) {
    case AV_CODEC_ID_H264:
        *coding = MPP_VIDEO_CodingAVC;
        *bsf_name = "h264_mp4toannexb";
        return 0;
    case AV_CODEC_ID_HEVC:
        *coding = MPP_VIDEO_CodingHEVC;
        *bsf_name = "hevc_mp4toannexb";
        return 0;
    default:
        return -1;
    }
}

vp_mpp_file_src_node::vp_mpp_file_src_node(std::string node_name,
                                          int channel_index,
                                          std::string file_path,
                                          bool cycle)
    : vp_src_node(node_name, channel_index, 1.0f),
      file_path(file_path),
      cycle(cycle) {
    VP_INFO(vp_utils::string_format("[%s] creating MPP source node for file: %s",
                               node_name.c_str(), file_path.c_str()));
    this->initialized();
}

vp_mpp_file_src_node::~vp_mpp_file_src_node() {
    cleanup();
    deinitialized();
}

std::string vp_mpp_file_src_node::to_string() {
    return file_path;
}

int vp_mpp_file_src_node::init_demuxer() {
    int ret;
    const char *bsf_name = NULL;
    const AVBitStreamFilter *bsf = NULL;
    char err[128];

    ret = avformat_open_input(&ifmt, file_path.c_str(), NULL, NULL);
    if (ret < 0) {
        av_strerror(ret, err, sizeof(err));
        VP_ERROR(vp_utils::string_format("[%s] avformat_open_input failed: %s",
                                   node_name.c_str(), err));
        return -1;
    }

    ret = avformat_find_stream_info(ifmt, NULL);
    if (ret < 0) {
        av_strerror(ret, err, sizeof(err));
        VP_ERROR(vp_utils::string_format("[%s] avformat_find_stream_info failed: %s",
                                   node_name.c_str(), err));
        return -1;
    }

    video_index = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_index < 0) {
        VP_ERROR(vp_utils::string_format("[%s] no video stream found", node_name.c_str()));
        return -1;
    }

    AVStream* vst = ifmt->streams[video_index];

    // Get FPS from stream
    AVRational fr = vst->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0)
        fr = vst->r_frame_rate;
    if (fr.num > 0 && fr.den > 0) {
        double stream_fps = av_q2d(fr);
        if (stream_fps > 1.0 && stream_fps < 240.0) {
            this->original_fps = (int)stream_fps;
        }
    }

    // Map codec and setup bitstream filter
    if (map_dec_codec(vst->codecpar->codec_id, &coding, &bsf_name)) {
        VP_ERROR(vp_utils::string_format("[%s] unsupported codec, only H.264/H.265 supported",
                                   node_name.c_str()));
        return -1;
    }

    bsf = av_bsf_get_by_name(bsf_name);
    if (!bsf) {
        VP_ERROR(vp_utils::string_format("[%s] failed to find BSF: %s",
                                   node_name.c_str(), bsf_name));
        return -1;
    }

    ret = av_bsf_alloc(bsf, &ibsfc);
    if (ret < 0)
        return -1;

    ret = avcodec_parameters_copy(ibsfc->par_in, vst->codecpar);
    if (ret < 0)
        return -1;

    ibsfc->time_base_in = vst->time_base;
    ret = av_bsf_init(ibsfc);
    if (ret < 0)
        return -1;

    VP_INFO(vp_utils::string_format("[%s] demuxer initialized, fps=%d, codec=%s",
                               node_name.c_str(), this->original_fps,
                               (coding == MPP_VIDEO_CodingAVC) ? "H264" : "H265"));

    return 0;
}

int vp_mpp_file_src_node::init_decoder() {
    // Create MPP decoder with callback
    decoder = std::make_unique<MppDecoder>((coding == MPP_VIDEO_CodingAVC) ? 264 : 265,
                                       this->original_fps, this,
                                       [](void* usrdata, int hor_stride, int ver_stride,
                                          int hor_width, int ver_height,
                                          int format, int fd, void* data) {
                                           auto node = static_cast<vp_mpp_file_src_node*>(usrdata);
                                           node->on_decoded_frame(hor_stride, ver_stride, hor_width,
                                                                ver_height, format, fd, data);
                                       });

    if (!decoder) {
        VP_ERROR(vp_utils::string_format("[%s] failed to allocate MPP decoder object",
                                   node_name.c_str()));
        return -1;
    }

    if (!decoder->mpp_ctx) {
        VP_ERROR(vp_utils::string_format("[%s] MPP decoder created but ctx is NULL (Init failed?)",
                                   node_name.c_str()));
        return -1;
    }

    VP_INFO(vp_utils::string_format("[%s] MPP decoder initialized successfully, ctx=%p",
                               node_name.c_str(), decoder->mpp_ctx));
    return 0;
}

void vp_mpp_file_src_node::on_decoded_frame(int hor_stride, int ver_stride,
                                          int hor_width, int ver_height,
                                          int format, int fd, void* data) {
    if (!got_first_frame) {
        width = hor_width;
        height = ver_height;
        stride_h = hor_stride;
        stride_v = ver_stride;
        original_width = hor_width;
        original_height = ver_height;
        got_first_frame = true;

        VP_INFO(vp_utils::string_format("[%s] first frame: w=%d h=%d stride=%d:%d",
                                   node_name.c_str(), width, height, stride_h, stride_v));
    }

    // Calculate NV12 buffer size
    size_t y_size = (size_t)stride_h * (size_t)stride_v;
    size_t uv_size = y_size / 2;
    size_t total_size = y_size + uv_size;

    // Create frame meta with NV12 data
    auto out_meta = std::make_shared<vp_objects::vp_frame_meta>(
        cv::Mat(),  // Empty Mat for compatibility
        this->frame_index,
        this->channel_index,
        width,
        height,
        this->original_fps
    );

    // Set NV12 format fields
    out_meta->is_nv12 = true;
    out_meta->dma_fd = fd;
    out_meta->nv12_data = data;
    out_meta->nv12_data_size = total_size;
    out_meta->stride_h = stride_h;
    out_meta->stride_v = stride_v;

    this->out_queue.push(out_meta);
    this->out_queue_semaphore.signal();

    VP_DEBUG(vp_utils::string_format("[%s] decoded frame %d", node_name.c_str(), frame_index));

    this->frame_index++;
}

void vp_mpp_file_src_node::cleanup() {
    demuxer_running = false;
    decoder_running = false;

    if (ibsfc) {
        av_bsf_free(&ibsfc);
        ibsfc = nullptr;
    }

    if (ifmt) {
        avformat_close_input(&ifmt);
        ifmt = nullptr;
    }

    decoder.reset();

    // Clear frame queue
    std::lock_guard<std::mutex> lock(frame_queue_lock);
    while (!frame_queue.empty()) {
        frame_queue.pop();
    }
}

void vp_mpp_file_src_node::handle_run() {
    if (init_demuxer() < 0) {
        VP_ERROR(vp_utils::string_format("[%s] failed to initialize demuxer",
                                   node_name.c_str()));
        // Send EOS to downstream
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();
        return;
    }

    if (init_decoder() < 0) {
        VP_ERROR(vp_utils::string_format("[%s] failed to initialize decoder",
                                   node_name.c_str()));
        // Send EOS to downstream
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();
        return;
    }

    demuxer_running = true;

    AVPacket* ipkt = av_packet_alloc();
    AVPacket* fpkt = av_packet_alloc();
    int ret = 0;

    if (!ipkt || !fpkt) {
        VP_ERROR(vp_utils::string_format("[%s] failed to allocate packets",
                                   node_name.c_str()));
        return;
    }

    VP_INFO(vp_utils::string_format("[%s] starting demuxer loop", node_name.c_str()));

    while (alive && gate.is_open() &&
           av_read_frame(ifmt, ipkt) >= 0) {
        if (ipkt->stream_index == video_index) {
            ret = av_bsf_send_packet(ibsfc, ipkt);
            if (ret < 0) {
                if (ret != AVERROR(EAGAIN)) {
                    VP_ERROR(vp_utils::string_format("[%s] bsf_send_packet failed",
                                               node_name.c_str()));
                    break;
                }
            }

            while (alive && (ret = av_bsf_receive_packet(ibsfc, fpkt)) >= 0) {
                // Send packet to MPP decoder in chunks
                RK_U8* data = fpkt->data;
                size_t total = (size_t)fpkt->size;
                size_t off = 0;

                while (off < total) {
                    size_t chunk = total - off;
                    if (chunk > kInputChunkSize)
                        chunk = kInputChunkSize;

                    decoder->Decode(data + off, chunk,
                                (off + chunk == total) ? 1 : 0);

                    off += chunk;
                }

                av_packet_unref(fpkt);
            }

            if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                VP_ERROR(vp_utils::string_format("[%s] bsf_receive_packet failed: %d",
                                           node_name.c_str(), ret));
                break;
            }
        }
        av_packet_unref(ipkt);

        // No delay for maximum throughput performance
    }

    // Flush decoder
    if (alive && gate.is_open()) {
        av_bsf_send_packet(ibsfc, NULL);
        for (;;) {
            ret = av_bsf_receive_packet(ibsfc, fpkt);
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
                break;
            if (ret < 0)
                break;

            decoder->Decode(fpkt->data, fpkt->size, 0);
            av_packet_unref(fpkt);
        }

        decoder->Decode(NULL, 0, 1);
    }

    av_packet_free(&ipkt);
    av_packet_free(&fpkt);

    // Send end-of-stream flag
    this->out_queue.push(nullptr);
    this->out_queue_semaphore.signal();

    // Mark as finished for main program to detect
    finished = true;

    VP_INFO(vp_utils::string_format("[%s] source node finished, total frames=%d",
                               node_name.c_str(), frame_index));
}

} // namespace vp_nodes
