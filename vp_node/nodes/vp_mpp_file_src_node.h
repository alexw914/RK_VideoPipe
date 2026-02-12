#pragma once

#include <string>
#include <queue>
#include <atomic>
#include <mutex>

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}

#include "mpp_decoder.h"
#include "nodes/base/vp_src_node.h"

namespace vp_nodes {
    // MPP hardware decoder source node
    // Uses FFmpeg for demuxing and MPP for hardware decoding
    // Outputs NV12 format directly for maximum performance
    class vp_mpp_file_src_node: public vp_src_node {
    private:
        std::string file_path;
        bool cycle;

        // FFmpeg components for demuxing
        AVFormatContext* ifmt = nullptr;
        AVBSFContext* ibsfc = nullptr;
        int video_index = -1;
        MppCodingType coding;

        // MPP decoder wrapper
        std::unique_ptr<MppDecoder> decoder;

        // Frame information from decoder
        int width = 0;
        int height = 0;
        int stride_h = 0;
        int stride_v = 0;
        bool got_first_frame = false;

        // Frame queue for decoded output
        std::queue<void*> frame_queue;
        std::mutex frame_queue_lock;

        // Control flags
        std::atomic<bool> demuxer_running{false};
        std::atomic<bool> decoder_running{false};

        // Initialize FFmpeg demuxer
        int init_demuxer();

        // Initialize MPP decoder
        int init_decoder();

        // Demuxer thread function
        void demuxer_thread_func();

        // Process decoded frame from MPP
        void on_decoded_frame(int hor_stride, int ver_stride, int hor_width,
                           int ver_height, int format, int fd, void* data);

        // Cleanup resources
        void cleanup();

    protected:
        virtual void handle_run() override;

    public:
        vp_mpp_file_src_node(std::string node_name,
                          int channel_index,
                          std::string file_path,
                          bool cycle = true);
        ~vp_mpp_file_src_node();

        virtual std::string to_string() override;

        // Flag to track when video source has finished
        std::atomic<bool> finished{false};
    };
}
