#pragma once
#include "rkbase.h"
#include "yolo.h"

/* RKPOSE img must fix shape [256x192]
   Give up AffineTransform, use LetterBox.
*/

class RTMPose:public RKBASE{
public:
    RTMPose(PoseConfig& config):RKBASE(config.model_path){
        spdlog::info("Model Type is RTMPose...");
    }
    static int load_config(const std::string &json_path, PoseConfig& conf);
    void run(const cv::Mat &src, KeyPointResult &res);
    void run(std::vector<cv::Mat> &img_datas, std::vector<KeyPointResult> &res_datas);
private:
    int postprocess(float *simcc_x_result, float *simcc_y_result, int extend_width, int extend_height, LETTER_BOX &lb, KeyPointResult &keypoint_result);
    void run_model(void* buf, LETTER_BOX &lb, KeyPointResult &result);
    void set_letterbox(int in_w, int in_h, LETTER_BOX &lb);
    void init_letterbox(LETTER_BOX &lb);
    PoseConfig config;
};