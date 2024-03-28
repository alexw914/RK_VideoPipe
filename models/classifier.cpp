#include "classifier.h"

void Classifier::run(const cv::Mat &src, std::vector<ClsResult> &res)
{
    unsigned char *resize_buf = (unsigned char *)malloc(model_width * model_height * 3);
    RESIZE_BOX rb;
    if (src.cols == model_width && src.rows == model_height)
    {
        run_model(src.data, res);
    }
    else{
        set_resizebox(src.cols, src.rows, rb);
        // ret = _resize_(src.data, resize_buf, rb);       // rga resize, may cause some error when use letterbox together.
        cv::Mat resized_img;
        cv::resize(src, resized_img, cv::Size(model_width, model_height));
        run_model(resized_img.data, res);
    }
    if (resize_buf) free(resize_buf);
}

// void Classifier::run(image_buffer_t &src, std::vector<ClsResult> &res)
// {
//     cv::Mat img(cv::Size(src.width, src.height), CV_8UC3, src.virt_addr);
//     run(img, res);
// }

void Classifier::run_model(void* buf, std::vector<ClsResult> &res)
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
    // Post Process
    softmax((float*)outputs[0].buf, output_attrs[0].n_elems);
    get_topk_with_indices((float*)outputs[0].buf, output_attrs[0].n_elems, res, config.topk);
    rknn_outputs_release(ctx, io_num.n_output, outputs);
}

void Classifier::set_resizebox(int width, int height, RESIZE_BOX &rb)
{
    rb.in_width      = width;
    rb.in_height     = height;
    rb.target_width  = model_width;
    rb.target_height = model_height;
}

void Classifier::softmax(float* array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }
    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }
    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }
    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

void Classifier::get_topk_with_indices(float* arr, int size, std::vector<ClsResult> &res, int k) {
    // 创建元素数组，保存值和索引号
    std::vector<element_t> elements;
    for (int i = 0; i < size; i++)
    {
        element_t elm;
        elm.value = *(arr+i);
        elm.index = i;
        elements.push_back(std::move(elm));
    }
    // 对元素数组进行快速排序
    std::sort(elements.begin(), elements.end(), [](element_t &elmt1, element_t &elmt2) -> bool
              { return elmt1.value > elmt2.value; }
    );
    // 获取前K个最大值和它们的索引号
    for (int i = 0; i < k; i++)
    {
        ClsResult cls;
        cls.score = elements[i].value;
        cls.label = config.labels.at(elements[i].index);
        res.push_back(cls);
    }
}
