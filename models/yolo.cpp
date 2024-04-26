#include "yolo.h"
#include <fstream>

int YOLO::load_config(const std::string &json_path, YOLOConfig& conf)
{
    std::ifstream f(json_path);
    json j_conf;
    f >> j_conf;
    conf.model_path = j_conf["model_path"].template get<std::string>();
    conf.conf_threshold = j_conf["conf_threshold"].template get<float>();
    conf.nms_threshold = j_conf["nms_threshold"].template get<float>();
    for (auto& it : j_conf["labels"]){
        conf.labels.push_back(it.template get<std::string>());
    }
    for (auto& it : j_conf["alarm_labels"]){
        conf.alarm_labels.push_back(it.template get<std::string>());
    }
    std::string model_type = j_conf["model_type"].template get<std::string>();
    if (model_type == "YOLOv5"){
        conf.type = ModelType::YOLOv5;
    }
    else if (model_type == "YOLOv6"){
        conf.type = ModelType::YOLOv6;
    }
    else if (model_type == "YOLOv7"){
        conf.type = ModelType::YOLOv7;
    }
    else if (model_type == "YOLOv8"){
        conf.type = ModelType::YOLOv8;
    }
    return 0;
}

void YOLO::run(const cv::Mat &src, std::vector<DetectionResult> &res)
{
//     unsigned char *resize_buf = (unsigned char *)malloc(model_width * model_height * 3);
//     LETTER_BOX lb;
//     init_letterbox(lb);
//     if (src.cols == model_width && src.rows == model_height)
//     {
//         run_model(src.data, lb, res);
//     }
//     else{
//         set_letterbox(src.cols, src.rows, lb);
//         lb.reverse_available = true;
//         ret = rga_letter_box_resize(src.data, resize_buf, &lb);
// #if 0 // debug rga resize
//         cv::Mat decode_img(cv::Size(model_width, model_height), CV_8UC3, resize_buf);
//         cv::cvtColor(decode_img, decode_img, cv::COLOR_RGB2BGR);
//         cv::imwrite("yolo_in.jpg", decode_img);
// #endif
//         if (ret == 0) run_model(resize_buf, lb, res);            // rga success!
//         else{                                                    // rga failed, run opencv!  
//         cv::Mat lb_img;
//         opencv_letter_box_resize(src, lb_img, lb);
//         run_model(lb_img.data, lb, res);
//         }
//     }
//     if (resize_buf) free(resize_buf);
    LETTER_BOX lb;
    init_letterbox(lb);
    if (src.cols == model_width && src.rows == model_height)
    {
        run_model(src.data, lb, res);
    }
    else{
        set_letterbox(src.cols, src.rows, lb);
        lb.reverse_available = true;
        cv::Mat lb_img;
        opencv_letter_box_resize(src, lb_img, lb);
        run_model(lb_img.data, lb, res);
    }
}

void YOLO::run(std::vector<cv::Mat> &img_datas, std::vector<std::vector<DetectionResult>> &res_datas){
    res_datas.clear();
    // scan 1 by 1
    for (int i = 0; i < img_datas.size(); i++) {
        std::vector<DetectionResult> res;
        this->run(img_datas[i], res);
        res_datas.push_back(res);
    }
}

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