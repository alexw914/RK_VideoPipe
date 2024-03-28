// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include "mpp_decoder.h"
#include "resize_function.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "im2d_common.h"

/*Include ffmpeg header file*/
#ifdef __cplusplus
extern "C" {
#endif
#include <libavformat/avformat.h>
#ifdef __cplusplus
};
#endif

typedef struct {
    int cam_idx;
    const char *rtsp_url = nullptr;
    int frame_width;
    int frame_height;
    int frame_rate;
    int frame_idx = 0;
    MppDecoder *decoder = nullptr;
    // YOLO *yolo = nullptr;            // Detection model
    // BYTETracker *tracker = nullptr;  // track
} app_context_t;

void callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data)
{
    app_context_t *ctx = (app_context_t *)userdata;
    int ret = 0;
    void *resize_buf = nullptr;
    resize_buf = malloc(width * height * 3);
    memset(resize_buf, 0, width * height * 3);
    if (ctx->frame_idx % 5 == 0){
        // yuv --> rgb, RGA
        rga_buffer_t src;
        rga_buffer_t dst;
        src = wrapbuffer_virtualaddr((void*)data, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
        dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);
        imcopy(src, dst);
        // debug
        cv::Mat decode_img(cv::Size(width, height), CV_8UC3, resize_buf);
        // run model ...
    }
    std::cout << "camera index: " << ctx->cam_idx << "frame index: " << std::to_string(ctx->frame_idx) << std::endl;
    ctx->frame_idx++;
    if (resize_buf) free(resize_buf);
}

/*-------------------------------------------
                  RTSP Functions
-------------------------------------------*/
void process_video_rtsp(void* userdata){

    app_context_t *ctx = (app_context_t *)userdata;

    AVFormatContext *pFormatCtx = NULL;
    AVDictionary *options = NULL;
    AVPacket *av_packet = NULL;

    avformat_network_init();
    av_dict_set(&options, "buffer_size", "1024000", 0); //设置缓存大小,1080p可将值跳到最大
    av_dict_set(&options, "rtsp_transport", "tcp", 0); //以tcp的方式打开,
    av_dict_set(&options, "stimeout", "5000000", 0); //设置超时断开链接时间，单位us
    av_dict_set(&options, "max_delay", "500000", 0); //设置最大时延

    pFormatCtx = avformat_alloc_context(); //用来申请AVFormatContext类型变量并初始化默认参数,申请的空间

    //打开网络流或文件流
    if (avformat_open_input(&pFormatCtx, ctx->rtsp_url, NULL, &options) != 0)
    {
        printf("Couldn't open input stream.\n");
        exit(0);
    }

    //获取视频文件信息
    if (avformat_find_stream_info(pFormatCtx, NULL)<0)
    {
        printf("Couldn't find stream information.\n");
        exit(0);
    }

    //查找码流中是否有视频流
    int videoindex = -1;
    unsigned i = 0;
    for (i = 0; i<pFormatCtx->nb_streams; i++)
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoindex = i;
            break;
        }
    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        exit(0);
    }

    av_packet = (AVPacket *)av_malloc(sizeof(AVPacket)); // 申请空间，存放的每一帧数据 （h264、h265)

    while (true)
    {
        if (av_read_frame(pFormatCtx, av_packet) >= 0)
        {
            if (av_packet->stream_index == videoindex)
            {
                // printf("--------------\ndata size is: %d\n-------------", av_packet->size);
                ctx->decoder->Decode(av_packet->data, av_packet->size, 0);
            }
            if (av_packet != NULL) av_packet_unref(av_packet);
        }
    }

    av_free(av_packet);
    avformat_close_input(&pFormatCtx);
}


int main(int argc, char** argv)
{
    std::ifstream in("assets/camera.txt");
    std::string s;
    std::vector<std::string> rtsp_urls;

    while (getline(in, s)) {
        rtsp_urls.push_back(s);
    }
    for(auto url: rtsp_urls){
        std::cout << url << std::endl;
    }
    int cam_num = rtsp_urls.size();

    std::thread threads[cam_num];

    app_context_t app_ctx[cam_num];

    for (auto idx = 0; idx < cam_num; idx++)
    {
        if (app_ctx[idx].decoder == nullptr) {
            auto decoder = new MppDecoder();
            decoder->Init(265, 25, &app_ctx[idx]);
            decoder->SetCallback(callback);
            app_ctx[idx].decoder = decoder;
        }

        app_ctx[idx].cam_idx = idx;
        app_ctx[idx].rtsp_url = rtsp_urls[idx].data();

        threads[idx] = std::thread(process_video_rtsp, &app_ctx[idx]);
    }

    for (auto& t: threads) {
        t.join();
    }

    return 0;
}