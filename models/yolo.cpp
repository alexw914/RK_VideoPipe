#include "yolo.h"

void YOLO::run(const cv::Mat& src, std::vector<DetectionResult>& res)
{
    unsigned char *resize_buf = (unsigned char *)malloc(model_width * model_height * 3);
    LETTER_BOX lb;
    init_letterbox(lb);
    if (src.cols == model_width && src.rows == model_height)
    {
        run_model(src.data, lb, res);
    }
    else{
        set_letterbox(src.cols, src.rows, lb);
        lb.reverse_available = true;
        ret = rga_letter_box_resize(src.data, resize_buf, &lb);
        #if 0 // debug rga resize
                cv::Mat decode_img(cv::Size(model_width, model_height), CV_8UC3, resize_buf);
                cv::imwrite("./resize.jpg", decode_img);
        #endif
        if (ret == 0) run_model(resize_buf, lb, res);            // rga success!
        else{                                                    // rga failed, run opencv!  
            cv::Mat lb_img;
            opencv_letter_box_resize(src, lb_img, lb);
            run_model(src.data, lb, res);
        }
    }
    if (resize_buf) free(resize_buf);
}

// void YOLO::run(image_buffer_t &src, std::vector<DetectionResult>& res)
// {   
//     // The buffer img format must RGB, such as the opencv Mat.
//     cv::Mat img(cv::Size(src.width, src.height), CV_8UC3, src.virt_addr);
//     run(img, res); 
// }

void YOLO::run_model(void* buf, LETTER_BOX &lb, std::vector<DetectionResult> &res)
{
    inputs[0].buf = buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx,  io_num.n_output, outputs, NULL);
    post->run(output_attrs, outputs, res, lb);
    rknn_outputs_release(ctx, io_num.n_output, outputs);
}

void YOLO::set_letterbox(int in_w, int in_h, LETTER_BOX& lb)
{
    lb.in_width = in_w;
    lb.in_height = in_h;
    lb.target_width = model_width;
    lb.target_height = model_height;
    compute_letter_box(&lb);
}

void YOLO::init_letterbox(LETTER_BOX &lb)
{
    memset(&lb, 0, sizeof(LETTER_BOX));
    lb.channel = 3;
    lb.target_width = model_width;
    lb.target_height = model_height;
}