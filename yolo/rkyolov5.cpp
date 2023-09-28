#include "rkyolov5.h"
#include "resize_function.h"
#include <sys/time.h>

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
double __get(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

RKYOLOV5::RKYOLOV5(char *model_name, int n, int frame_width, int frame_height)
{
    /* Create the neural network */
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // set core mask
    rknn_core_mask core_mask;
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if(n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    int ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    // vesrion
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // Get params
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // input_tensor
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // output tensor
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("Model is NCHW input format\n");
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("Model is NHWC input format\n");
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    printf("Model input height=%d, width=%d, channel=%d\n", model_height, model_width, model_channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;


    lb.target_height = model_height;
    lb.target_width = model_width;
    lb.in_height = frame_height;
    lb.in_width = frame_width;

    compute_letter_box(&lb);
}

RKYOLOV5::~RKYOLOV5()
{
    ret = rknn_destroy(ctx);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}

void RKYOLOV5::draw_box(cv::Mat& orig_img){

    char text[32];
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_group.results[i].box.left = w_reverse(detect_result_group.results[i].box.left, lb);
        detect_result_group.results[i].box.right = w_reverse(detect_result_group.results[i].box.right, lb);
        detect_result_group.results[i].box.top = h_reverse(detect_result_group.results[i].box.top, lb);
        detect_result_group.results[i].box.bottom = h_reverse(detect_result_group.results[i].box.bottom, lb);
        detect_result_t* det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
            det_result->box.right, det_result->box.bottom, det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::cvtColor(orig_img, orig_img, cv::COLOR_BGR2RGB);
    cv::imwrite("out.jpg", orig_img);

}

void RKYOLOV5::predict(void* input_data)
{

        // Letter box resize
    unsigned char *resize_buf = (unsigned char *)malloc(model_width * model_height * 3);

    if ((lb.in_height == lb.target_height) && (lb.in_width == lb.target_width)){
        inputs[0].buf = input_data;
    }
    else{
        ret = rga_letter_box_resize(input_data, resize_buf, &lb);
        lb.reverse_available = true;
        inputs[0].buf = resize_buf;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
        outputs[i].want_float = 0;
    
    // output
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx,  io_num.n_output, outputs, NULL);

    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;
    for (int i = 0; i < io_num.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
            model_height, model_width, 
            out_zps, out_scales, 
            &detect_result_group
    );

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    if (resize_buf){
        free(resize_buf);
    }
}


static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}