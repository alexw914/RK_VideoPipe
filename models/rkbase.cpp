#include "rkbase.h"

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
double __get(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
    spdlog::info("index={}, name={}, n_dims={}, dims=[{}], n_elems={}, size={}, w_stride={}, size_with_stride={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
            attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
            attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

RKBASE::RKBASE(const std::string& model_path, int n)
{
    /* Create the neural network */
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] [thread %t] %v");
    spdlog::info("Loading model..");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // set core mask
    rknn_core_mask core_mask;
    switch(n % 3){
        case 0:
            core_mask = RKNN_NPU_CORE_0;
        case 1:
            core_mask = RKNN_NPU_CORE_1;
        case 2:
            core_mask = RKNN_NPU_CORE_2;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        spdlog::error("rknn_init core error ret={}", ret);
        exit(-1);
    }

    // vesrion
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        spdlog::error("rknn_init error ret={}", ret);
        exit(-1);
    }

    // Get params
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        spdlog::error("rknn_init error ret={}", ret);
        exit(-1);
    }

    // input_tensor
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    spdlog::info("Input Tensors are as follows...");

    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // output tensor
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    spdlog::info("Output Tensors are as follows...");
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        spdlog::info("Model is NCHW input format...");
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else
    {
        spdlog::info("Model is NHWC input format...");
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
}

RKBASE::~RKBASE()
{
    ret = rknn_destroy(ctx);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
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

