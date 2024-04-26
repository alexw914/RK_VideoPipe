#pragma once
#include <vector>
#include <string>

enum class ModelType{YOLOv5, YOLOv6, YOLOv7, YOLOv8, ResNet18, MobileNetv3, RTMPose};

struct Config{
    std::string model_path;
    std::vector<std::string> labels;
    std::vector<std::string> alarm_labels;
    int core_mask = 0;
};

struct YOLOConfig: public Config{
    ModelType type = ModelType::YOLOv5;
    float conf_threshold = 0.25;
    float nms_threshold = 0.45;
};

struct ClsConfig: public Config{
    ModelType type = ModelType::ResNet18;
    int topk = 1;
};

struct PoseConfig: public Config{
    int keypoint_num = 17;
    ModelType type = ModelType::RTMPose;
};

struct BOX{
    int top = 0;
    int left = 0;
    int bottom = 0;
    int right = 0;
};

struct DetectionResult{
    BOX box;
    float score;
    int id;
    std::string label;
};

struct KeyPointResult {
    std::vector<std::pair<int, int>> keypoints;
    std::vector<float> scores;
};

struct ClsResult{
    int id;
    std::string label;
    float score;
};