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
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"
#include "yolo/rkyolov5.h"
#include "utils/mpp_decoder.h"
#include "player/player.h"

#define LOGD 
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

int main(int argc, char** argv)
{
    char* rtsp_url = "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/501";// rtsp 地址
    char *model_name = NULL;

    if (argc != 2)
    {
      printf("Usage: %s <rknn model>\n", argv[0]);
      return -1;
    }

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
        decoder->SetCallback(_frame_callback);
        app_ctx.decoder = decoder;
    }
    process_video_rtsp(&app_ctx, rtsp_url);

    return 0;
}

