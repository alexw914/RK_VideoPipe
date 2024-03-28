#pragma once
#include <iostream>
#include <sstream>
#include <set>

#include <vector>
#include <stdint.h>
#include "rkbase.h"
#include "spdlog/spdlog.h"

class PostProcessorBase{
public:
    PostProcessorBase() {}
    PostProcessorBase(const YOLOConfig& config) {
        this->config = config;
        std::string type_name;
        switch (config.type)
        {
        case ModelType::YOLOv5:
            type_name = std::string("YOLOv5");
            break;
        case ModelType::YOLOv6:
            type_name = std::string("YOLOv6");
            break;
        case ModelType::YOLOv7:
            type_name = std::string("YOLOv7");
            break;
        case ModelType::YOLOv8:
            type_name = std::string("YOLOv8");
            break;
        default:
            break;
        }
        std::stringstream lss;
        for (auto i = 0; i < config.labels.size(); i++){
            lss << config.labels[i];
            if (i != config.labels.size() - 1) lss << ", ";
        }
        spdlog::info("Post processor type is {}, labels: [{}], conf thresh: {}, nms thresh: {}.", type_name, lss.str(), config.conf_threshold, config.nms_threshold);
    }

    virtual int process_func(rknn_tensor_attr* output_attrs, rknn_output* outputs, int model_in_w, int model_in_h,
                        std::vector<float>& filterBoxes, std::vector<float>& objProbs, std::vector<int>& classId) { return 0; }

    int run(rknn_tensor_attr *output_attrs, rknn_output *outputs, std::vector<DetectionResult> &results, LETTER_BOX &lb);

protected:
    YOLOConfig config;
};

class YOLOv5v7_Post:public PostProcessorBase{
public:
    explicit YOLOv5v7_Post(YOLOConfig& config) : PostProcessorBase(config){}
    int process_i8(int8_t *input, int *anchor, int grid_h, int grid_w, int stride,
                    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, 
                    int32_t zp, float scale);

    virtual int process_func(rknn_tensor_attr *output_attrs, rknn_output *outputs, int model_in_w, int model_in_h,
                        std::vector<float> &filterBoxes, std::vector<float> &objProbs, std::vector<int> &classId) {
        int validCount = 0;
        int stride, grid_h, grid_w = 0;
        for (int i = 0; i < 3; i++){
            grid_h = output_attrs[i].dims[2];
            grid_w = output_attrs[i].dims[3];
            stride = model_in_h / grid_h;
            switch (config.type)
            {
            case ModelType::YOLOv5:
                validCount += process_i8((int8_t *)outputs[i].buf, (int *)anchor_yolov5[i], grid_h, grid_w, stride, 
                                        filterBoxes, objProbs, classId, 
                                        output_attrs[i].zp, output_attrs[i].scale);
                break;
            case ModelType::YOLOv7:
                validCount += process_i8((int8_t *)outputs[i].buf, (int *)anchor_yolov7[i], grid_h, grid_w, stride, 
                        filterBoxes, objProbs, classId, 
                        output_attrs[i].zp, output_attrs[i].scale);
            default:
                break;
            }
        }
        return validCount;
    }
private:
    const int anchor_yolov5[3][6] = {{10, 13, 16, 30, 33, 23},
                        {30, 61, 62, 45, 59, 119},
                        {116, 90, 156, 198, 373, 326}};
    const int anchor_yolov7[3][6] = {{12, 16, 19, 36, 40, 28},
                        {36, 75, 76, 55, 72, 146},
                        {142, 110, 192, 243, 459, 401}};
};


class YOLOv6v8_Post:public PostProcessorBase{
public:
    explicit YOLOv6v8_Post(YOLOConfig& config):PostProcessorBase(config){}
    
    int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                   int8_t *score_tensor, int32_t score_zp, float score_scale,
                   int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                   int grid_h, int grid_w, int stride, int dfl_len,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId);

    virtual int process_func(rknn_tensor_attr *output_attrs, rknn_output *outputs, int model_in_w, int model_in_h,
                        std::vector<float> &filterBoxes, std::vector<float> &objProbs, std::vector<int> &classId) {
        int validCount = 0;
        int stride, grid_h, grid_w = 0;
        int dfl_len = output_attrs[0].dims[1] / 4;
        int output_per_branch = 3;
        for (int i = 0; i < 3; i++)
        {
            void *score_sum = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            if (output_per_branch == 3){
                score_sum = outputs[i*output_per_branch + 2].buf;
                score_sum_zp = output_attrs[i*output_per_branch + 2].zp;
                score_sum_scale = output_attrs[i*output_per_branch + 2].scale;
            }
            int box_idx = i*output_per_branch;
            int score_idx = i*output_per_branch + 1;

            grid_h = output_attrs[box_idx].dims[2];
            grid_w = output_attrs[box_idx].dims[3];
            stride = model_in_h / grid_h;

            validCount += process_i8((int8_t *)outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                    (int8_t *)outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                    (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                    grid_h, grid_w, stride, dfl_len, 
                                    filterBoxes, objProbs, classId);
        }
        return validCount;
    }
};

