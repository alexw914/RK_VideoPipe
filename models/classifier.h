#include "rkbase.h"

class Classifier:public RKBASE{
public:
    typedef struct {
        float value;
        int index;
    } element_t;
    explicit Classifier(ClsConfig& config):RKBASE(config.model_path, config.core_mask) { 
        this->config = config;
    }
    void run(const cv::Mat &src, std::vector<ClsResult> &res);
    // void run(image_buffer_t& src, std::vector<ClsResult> &res);
    void run_model(void *buf, std::vector<ClsResult> &res);
private:
    void softmax(float *array, int size);
    void get_topk_with_indices(float* arr, int size, std::vector<ClsResult> &res, int k=1);
    void set_resizebox(int width, int height, RESIZE_BOX &rb);
    ClsConfig config;
};