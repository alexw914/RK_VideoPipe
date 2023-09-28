#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 5

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

class PostProcessor{
public:
    int process(int8_t *input, int *anchor,
                int grid_h, int grid_w, int stride,
                std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                float threshold,
                int32_t zp, float scale);

    int run(int8_t *input0, int8_t *input1, int8_t *input2,
            int model_in_h, int model_in_w,
            std::vector<int32_t> &qnt_zps,
            std::vector<float> &qnt_scales,
            detect_result_group_t *group);

    void set_class_num(const int obj_class_num, char *label_name);
    short int get_class_num() const { return OBJ_CLASS_NUM; }

    void set_conf_threshold(const float& conf_threshold) {
        conf_threshold_ = conf_threshold;
    }
    float get_conf_threshold() const { return conf_threshold_; }

    void set_nms_threshold(const float& nms_threshold) {
        nms_threshold_ = nms_threshold;
    }
    float get_nms_threshold() const { return nms_threshold_; }

    PostProcessor(){};
    PostProcessor(const int obj_class_num, char *label_name,
                  const float &conf_threshold, const float &nms_threshold);
    ~PostProcessor();

private:
    float conf_threshold_ = 0.25;
    float nms_threshold_ = 0.45;
    short int OBJ_CLASS_NUM = 2;
    short int PROP_BOX_SIZE = OBJ_CLASS_NUM + 5;
    // char labels[OBJ_NUMB_MAX_SIZE][OBJ_NAME_MAX_SIZE];
    const int anchor0[6] = {10, 13, 16, 30, 33, 23};
    const int anchor1[6] = {30, 61, 62, 45, 59, 119};
    const int anchor2[6] = {116, 90, 156, 198, 373, 326};
    char *labels[OBJ_NUMB_MAX_SIZE];
};