#include "rkbase.h"

class Classifier:public RKBASE{
public:
    typedef struct {
        float value;
        int index;
    } element_t;
    static int load_config(const std::string &json_path, ClsConfig& conf);
    explicit Classifier(ClsConfig& config):RKBASE(config.model_path) { 
        this->config = config;
    }
    void run(const cv::Mat &src, ClsResult &res);
    void run(std::vector<cv::Mat> &img_datas, std::vector<ClsResult> &res_datas);
    void run_model(void *buf, ClsResult &res);
private:
    void softmax(float *array, int size);
    void get_topk_with_indices(float* arr, int size, ClsResult &res);
    void set_resizebox(int width, int height, RESIZE_BOX &rb);
    ClsConfig config;
};