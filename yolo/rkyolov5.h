# pragma once

#include <queue>
#include <vector>
#include <iostream>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocessor.h"
#include "resize_function.h"

class RKYOLOV5
{
public:
    detect_result_group_t detect_result_group;
    PostProcessor post;

    LETTER_BOX lb;
    RKYOLOV5() {}
    RKYOLOV5(char *model_name, int n, int frame_width, int frame_height);

    void set_frame_size(const int frame_width, const int frame_height ){
        this->frame_height = frame_height;
        this->frame_width = frame_width;
    }
    
    void predict(void* input_data);
    void draw_box(cv::Mat& orig_img);
    ~RKYOLOV5();

private:

    int model_channel = 3;
    int model_width = 640;
    int model_height = 384;
    int frame_width = 1920;
    int frame_height = 1080;

    rknn_context ctx;
    unsigned char *model_data;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    int ret;

};