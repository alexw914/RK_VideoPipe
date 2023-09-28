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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "yolo/rkyolov5.h"
#include "yolo/postprocessor.h"
#include "yolo/resize_function.h"

#include "include/timer.h"

/*Get Stream using gstreamer backend*/

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{

    TIMER timer;
    timer.indent_set("    ");

    char *model_name = NULL;

    if (argc != 2)
    {
      printf("Usage: %s <rknn model>\n", argv[0]);
      return -1;
    }

    model_name       = (char*)argv[1];

    std::string rtsp_url = "rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/201";
    std::stringstream gss, pss;
    size_t framerate = 25;
    gss << "rtspsrc location=" << rtsp_url << " latency=0"
        << " ! rtph265depay ! h265parse ! mppvideodec "
        << " ! video/x-raw, framerate=" << framerate << "/1"
        << " ! videoconvert ! video/x-raw, format=BGR"
        << " ! appsink sync=false ";

    cv::VideoCapture cap;
    cap.open(gss.str(), cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cout << "Cannot open rtsp!" << std::endl;
        return 0;
    }


    std::string push_url = "rtmp://192.168.3.142:1935/live/123";
    pss << "appsrc ! videoconvert"
        << " ! x264enc ! flvmux streamable = true"
        << " ! rtmpsink location=" << push_url;
    cv::VideoWriter writer;
    writer.open(pss.str(),
        (int)cap.get(cv::CAP_PROP_FOURCC),
        (int)cap.get(cv::CAP_PROP_FPS),
        cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        , true // colorfull pic)
    );
    
    auto rkyolov5 = new RKYOLOV5(model_name, 0, 
                                (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    char* label_name = "./model/helmet_labels_list.txt";
    rkyolov5->post.set_class_num(2, label_name);


    cv::Mat frame;
    int frameNums = 0;

    while (cap.isOpened())
    {
        cap.read(frame);
        // cv::imwrite("out.jpg", frame);
        timer.tik();
        rkyolov5->predict((void*)frame.data);
        timer.tok();
        timer.print_time("infer");

        // if (rkyolov5->detect_result_group.count > 0)
        //     rkyolov5->draw_box((void*)frame.data);
        frameNums++;
    }

    cap.release();
    return 0;
}
