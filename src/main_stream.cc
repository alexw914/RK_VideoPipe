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
#include "opencv2/core/core.hpp"
#include <opencv2/videoio.hpp>
#include "yolo/rkyolov5.h"
#include "yolo/postprocessor.h"
#include "utils/mpp_decoder.h"
#include "include/timer.h"

/*Include ffmpeg header file*/
#ifdef __cplusplus
extern "C" {
#endif
#include <libavformat/avformat.h>
#ifdef __cplusplus
};
#endif
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
typedef struct {
    RKYOLOV5* rkyolov5;
    MppDecoder* decoder;
} app_context_t;

void mpp_decoder_frame_callback(void *userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void *data)
{

    TIMER timer;
    timer.indent_set("    ");    
    app_context_t* ctx = (app_context_t*)userdata;

    int ret = 0;
    void *resize_buf = nullptr;
    resize_buf = malloc(width * height * 3);
    memset(resize_buf, 0, width * height * 3);

    rga_buffer_t src;
    rga_buffer_t dst;
    src = wrapbuffer_virtualaddr((void*)data, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
    dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);
    imcopy(src, dst);
    // // for debug

    cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);

    // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    // cv::imwrite("resize_input.jpg", resize_img);

    timer.tik();
    ctx->rkyolov5->predict((void*)resize_buf);
    timer.tok();
    timer.print_time("predict");
    // timer.tik();
    // ctx->rkyolov5->draw_box(resize_img);
    // timer.tok();
    // timer.print_time("save picture");

    if (resize_buf) free(resize_buf);
}

int main(int argc, char** argv)
{
    char *model_name = NULL;

    if (argc != 2)
    {
      printf("Usage: %s <rknn model>\n", argv[0]);
      return -1;
    }


    AVFormatContext *pFormatCtx = NULL;
    AVDictionary *options = NULL;
    AVPacket *av_packet = NULL;
    char filepath[] = "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/101";// rtsp 地址

    // char filepath[] = "rtmp://192.168.3.228/live/test1";
    av_register_all(); // 函数在ffmpeg4.0以上版本已经被废弃，所以4.0以下版本就需要注册初始函数
    avformat_network_init();
    av_dict_set(&options, "buffer_size", "1024000", 0); //设置缓存大小,1080p可将值跳到最大
    av_dict_set(&options, "rtsp_transport", "tcp", 0); //以tcp的方式打开,
    av_dict_set(&options, "stimeout", "5000000", 0); //设置超时断开链接时间，单位us
    av_dict_set(&options, "max_delay", "500000", 0); //设置最大时延

    pFormatCtx = avformat_alloc_context(); //用来申请AVFormatContext类型变量并初始化默认参数,申请的空间


    //打开网络流或文件流
    if (avformat_open_input(&pFormatCtx, filepath, NULL, &options) != 0)
    {
        printf("Couldn't open input stream.\n");
        return 0;
    }

    //获取视频文件信息
    if (avformat_find_stream_info(pFormatCtx, NULL)<0)
    {
        printf("Couldn't find stream information.\n");
        return 0;
    }

    //查找码流中是否有视频流
    int videoindex = -1;
    unsigned i = 0;
    for (i = 0; i<pFormatCtx->nb_streams; i++)
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoindex = i;
            break;
        }
    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        return 0;
    }

    av_packet = (AVPacket *)av_malloc(sizeof(AVPacket)); // 申请空间，存放的每一帧数据 （h264、h265）

    //这边可以调整i的大小来改变文件中的视频时间,每 +1 就是一帧数据

    app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_context_t));
    model_name       = (char*)argv[1];

    if (app_ctx.rkyolov5 == nullptr){
        auto rkyolov5 = new RKYOLOV5(model_name, 0, 1920, 1080);
        char *label_name = "./model/helmet_labels_list.txt";
        rkyolov5->post.set_class_num(2, label_name);
        app_ctx.rkyolov5 = rkyolov5;
    }

    if (app_ctx.decoder == NULL) {
        auto decoder = new MppDecoder();
        decoder->Init(265, 20, &app_ctx);
        decoder->SetCallback(mpp_decoder_frame_callback);
        app_ctx.decoder = decoder;
    }

    int count = 1;
    int step = 5;
    while (true)
    {
        if (av_read_frame(pFormatCtx, av_packet) >= 0)
        {
            if (av_packet->stream_index == videoindex)
            {
                // printf("--------------\ndata size is: %d\n-------------", av_packet->size);
                if (count % step == 0){
                    app_ctx.decoder->Decode(av_packet->data, av_packet->size, 0);
                    count = 1;
                }
            }
            if (av_packet != NULL)
                av_packet_unref(av_packet);
        }
        count++;
    }

//    fclose(fpSave);
    av_free(av_packet);
    avformat_close_input(&pFormatCtx);
    return 0;
}