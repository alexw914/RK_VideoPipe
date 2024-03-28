#pragma once
#include "rkbase.h"
#include "yolo.h"
// #include "BYTETracker.h"

/* RKPOSE img must fix shape [256x192]*/
class RTMPose:public RKBASE{
public:
    RTMPose(PoseConfig& config):RKBASE(config.model_path, config.core_mask){
        spdlog::info("Model Type is RTMPose...");
    }
    void run(const cv::Mat &src, KeyPointDetectionResult &result);
    int postprocess(float *simcc_x_result, float *simcc_y_result, int extend_width, int extend_height, KeyPointDetectionResult &keypoint_result);
    int get_keypoint_num() { return config.keypoint_num; }
private:
    PoseConfig config;
};

/*
    if give one raw picture, just use this class.
*/
class RTMPoseTracker{
public:
    RTMPoseTracker(YOLOConfig& y_config, PoseConfig& p_config){
        this->yolo = new YOLO(y_config);
        this->rtmpose = new RTMPose(p_config);
    }
    ~RTMPoseTracker(){
        if (yolo) free(yolo);
        if (rtmpose) free(rtmpose);
    }
    void run(const cv::Mat &src, std::vector<DetectionResult>& det_res, std::vector<KeyPointDetectionResult>& kpt_res);
    // void run(image_buffer_t& src, std::vector<DetectionResult>& det_res, std::vector<KeyPointDetectionResult>& kpt_res);
private:
    YOLO* yolo = nullptr;
    RTMPose* rtmpose = nullptr;
    // BYTETracker *tracker = nullptr;
};


