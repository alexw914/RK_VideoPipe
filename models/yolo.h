#pragma once
#include "rkbase.h"
#include "yolo_post.h"

// Object detection Task, support: yolov5 to yolov8.
class YOLO:public RKBASE{
public:
    explicit YOLO(YOLOConfig& config):RKBASE(config.model_path, config.core_mask){
        switch(config.type){
        case ModelType::YOLOv5:
            post = new YOLOv5v7_Post(config);
            break;
        case ModelType::YOLOv6:
            post = new YOLOv6v8_Post(config);
            break;
        case ModelType::YOLOv7:
            post = new YOLOv5v7_Post(config);
            break;
        case ModelType::YOLOv8:
            post = new YOLOv6v8_Post(config);
            break;
        default:
            break;
        }
    }
    ~YOLO() {
        if (post) free(post);
        spdlog::info("Free RKYOLO Model...");
    }
    void run(const cv::Mat& src, std::vector<DetectionResult> &res);
    // void run(image_buffer_t& src, std::vector<DetectionResult> &res);
    void run_model(void* buf, LETTER_BOX &lb, std::vector<DetectionResult> &res);
private:
    void set_letterbox(int in_w, int in_h, LETTER_BOX &lb);
    void init_letterbox(LETTER_BOX &lb);
    PostProcessorBase *post = nullptr;
};