#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "resize_function.h"


#define ENABLE_RGA

#ifdef ENABLE_RGA
#include "im2d.h"
#include "RgaUtils.h"
#include "im2d_common.h"
// #include "utils.h"
// #include "dma_alloc.h"
#endif

int compute_letter_box(LETTER_BOX* lb){
    lb->img_wh_ratio = (float)lb->in_width/ (float)lb->in_height;
    lb->target_wh_ratio = (float)lb->target_width/ (float)lb->target_height;

    if (lb->img_wh_ratio >= lb->target_wh_ratio){
        //pad height dim
        lb->resize_scale_w = (float)lb->target_width / (float)lb->in_width;
        lb->resize_scale_h = lb->resize_scale_w;

        lb->resize_width = lb->target_width;
        lb->w_pad_left = 0;
        lb->w_pad_right = 0;

        lb->resize_height = (int)((float)lb->in_height * lb->resize_scale_h);
        lb->h_pad_top = (lb->target_height - lb->resize_height) / 2;
        if (((lb->target_height - lb->resize_height) % 2) == 0){
            lb->h_pad_bottom = lb->h_pad_top;
        }
        else{
            lb->h_pad_bottom = lb->h_pad_top + 1;
        }

    }
    else{
        //pad width dim
        lb->resize_scale_h = (float)lb->target_height / (float)lb->in_height;
        lb->resize_scale_w = lb->resize_scale_h;

        lb->resize_width = (int)((float)lb->in_width * lb->resize_scale_w);
        lb->w_pad_left = (lb->target_width - lb->resize_width) / 2;
        if (((lb->target_width - lb->resize_width) % 2) == 0){
            lb->w_pad_right = lb->w_pad_left;
        }
        else{
            lb->w_pad_right = lb->w_pad_left + 1;
        }        

        lb->resize_height = lb->target_height;
        lb->h_pad_top = 0;
        lb->h_pad_bottom = 0;
    }
    return 0;
}

#ifdef ENABLE_RGA
int _rga_resize(rga_buffer_handle_t src_handle, rga_buffer_handle_t dst_handle, LETTER_BOX* lb){
    int ret = 0;
    rga_buffer_t src_buf, dst_buf;

    src_buf = wrapbuffer_handle(src_handle, lb->in_width, lb->in_height, RK_FORMAT_RGB_888);
    dst_buf = wrapbuffer_handle(dst_handle, lb->target_width, lb->target_height, RK_FORMAT_RGB_888);

    im_rect dst_rect;
    dst_rect.x = 0 + lb->w_pad_left;
    dst_rect.y = 0 + lb->h_pad_top;
    dst_rect.width = lb->resize_width;
    dst_rect.height = lb->resize_height;

    ret = imcheck(src_buf, dst_buf, {}, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        printf("rga letter box resize check error! \n");
        return -1;
    }

    ret = improcess(src_buf, dst_buf, {}, {}, dst_rect, {}, IM_SYNC);
    if (ret == IM_STATUS_SUCCESS) {
        // printf("%s running success!\n", "rga letter box resize");
    } else {
        printf("%s running failed, %s\n", "rga letter box resize", imStrError((IM_STATUS)ret));
        return -1;
    }
    return 0;
}
#endif

int rga_letter_box_resize(void* src_buf, void* dst_buf, LETTER_BOX* lb){
    int ret = 0;
#ifdef ENABLE_RGA
    memset(dst_buf, 114, lb->target_width * lb->target_height * lb->channel);
    rga_buffer_handle_t src_handle, dst_handle;
    src_handle = importbuffer_virtualaddr(src_buf, lb->in_width* lb->in_height* lb->channel);
    dst_handle = importbuffer_virtualaddr(dst_buf, lb->target_width* lb->target_height* lb->channel);

    ret = _rga_resize(src_handle, dst_handle, lb);

    if (src_handle > 0){
        releasebuffer_handle(src_handle);}
    if (dst_handle > 0){
        releasebuffer_handle(dst_handle);}
#else
    ret = -1;
#endif
    return ret;
}


int rga_letter_box_resize(void *src_buf, int dst_fd, LETTER_BOX* lb){
    int ret = 0;

#ifdef ENABLE_RGA
    rga_buffer_handle_t src_handle, dst_handle;

#ifdef RV110X_DEMO
    int src_fd;
    char* tmp_buf;
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, lb->in_width* lb->in_height* lb->channel, &src_fd, (void **)&tmp_buf);
    memcpy(tmp_buf, src_buf, lb->in_width* lb->in_height* lb->channel);
    src_handle = importbuffer_fd(src_fd, lb->in_width* lb->in_height* lb->channel);
#else
    src_handle = importbuffer_virtualaddr(src_buf, lb->in_width* lb->in_height* lb->channel);
#endif

    dst_handle = importbuffer_fd(dst_fd, lb->target_width* lb->target_height* lb->channel);

    ret = _rga_resize(src_handle, dst_handle, lb);

    if (src_handle > 0){
        releasebuffer_handle(src_handle);}
    if (dst_handle > 0){
        releasebuffer_handle(dst_handle);}

#ifdef RV110X_DEMO
    dma_buf_free(lb->in_width* lb->in_height* lb->channel, &src_fd, tmp_buf);
#endif
#else
    ret = -1;
#endif
    return ret;
}

int rga_letter_box_resize(int src_fd, int dst_fd, LETTER_BOX* lb){
    int ret = 0;

#ifdef ENABLE_RGA
    rga_buffer_handle_t src_handle, dst_handle;

    src_handle = importbuffer_fd(src_fd, lb->in_width* lb->in_height* lb->channel);
    dst_handle = importbuffer_fd(dst_fd, lb->target_width* lb->target_height* lb->channel);

    ret = _rga_resize(src_handle, dst_handle, lb);

    if (src_handle > 0){
        releasebuffer_handle(src_handle);}
    if (dst_handle > 0){
        releasebuffer_handle(dst_handle);}
#else
    ret = -1;
#endif

    return ret;
}

int letter_box_resize(void *src_buf, void *dst_buf, LETTER_BOX &lb, char pad_color){
    int ret = 0;
#ifdef ENABLE_RGA
    memset(dst_buf, 114, lb.target_width * lb.target_height * lb.channel);
    ret = rga_letter_box_resize(src_buf, dst_buf, &lb);
#else
    ret = -1;
#endif
    if (ret != 0) cpu_letter_box_resize(src_buf, dst_buf, lb);
    return ret;
}

void opencv_letter_box_resize(const cv::Mat &image, cv::Mat &padded_image, LETTER_BOX& lb, const cv::Scalar &pad_color)
{
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(lb.resize_width, lb.resize_height));
    // 在图像周围添加填充
    cv::copyMakeBorder(resized_image, padded_image, lb.h_pad_top, lb.h_pad_bottom, lb.w_pad_left, lb.w_pad_right, cv::BORDER_CONSTANT, pad_color);
}

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

int h_reverse(int h, LETTER_BOX& lb){
    if (lb.reverse_available == false) return h;
    int r_h = clamp(h, 0, lb.target_height) - lb.h_pad_top;
    r_h = clamp(r_h, 0, lb.target_height);
    r_h = r_h / lb.resize_scale_h;
    return r_h;
}

int w_reverse(int w, LETTER_BOX& lb){
    if (lb.reverse_available == false) return w;
    int r_w = clamp(w, 0, lb.target_width) - lb.w_pad_left;
    r_w = clamp(r_w, 0, lb.target_width);
    r_w = r_w / lb.resize_scale_w;
    return r_w;
}

int print_letter_box_info(LETTER_BOX& lb){
    printf("in_width: %d\n", lb.in_width);
    printf("in_height: %d\n", lb.in_height);
    printf("target_width: %d\n", lb.target_width);
    printf("target_height: %d\n", lb.target_height);
    printf("img_wh_ratio: %f\n", lb.img_wh_ratio);
    printf("target_wh_ratio: %f\n", lb.target_wh_ratio);
    printf("resize_scale_w: %f\n", lb.resize_scale_w);
    printf("resize_scale_h: %f\n", lb.resize_scale_h);
    printf("resize_width: %d\n", lb.resize_width);
    printf("resize_height: %d\n", lb.resize_height);
    printf("w_pad_left: %d\n", lb.w_pad_left);
    printf("w_pad_right: %d\n", lb.w_pad_right);
    printf("h_pad_top: %d\n", lb.h_pad_top);
    printf("h_pad_bottom: %d\n", lb.h_pad_bottom);
    printf("reverse_available: %d\n", lb.reverse_available);
    return 0;
}

int cpu_letter_box_resize(void *src_buf, void *dst_buf, LETTER_BOX &lb, char pad_color)
{
    // Only support RGB--->RGB
    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = lb.in_width;
    int src_box_h = lb.in_height;

    int dst_box_x = lb.w_pad_left;
    int dst_box_y = lb.h_pad_top;
    int dst_box_w = lb.target_width - lb.w_pad_right - lb.w_pad_left;
    int dst_box_h = lb.target_height - lb.h_pad_bottom - lb.h_pad_top;

    int reti = 0;
    reti = crop_and_scale_image_c(3, (unsigned char*)src_buf, lb.in_width, lb.in_height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            (unsigned char*)dst_buf, lb.target_width, lb.target_height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);

    if (reti != 0) {
        printf("Convert image by cpu failed %d\n", reti);
        return -1;
    }
    return 0;
}

int rga_resize(void* src_buf, void* dst_buf, RESIZE_BOX &rb)
{
    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;
    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_handle = importbuffer_virtualaddr(src_buf, rb.in_height * rb.in_width * 3);
    dst_handle = importbuffer_virtualaddr(dst_buf, rb.target_height * rb.target_width * 3);
    src_img = wrapbuffer_handle(src_handle, rb.in_width, rb.in_height, RK_FORMAT_RGB_888);
    dst_img = wrapbuffer_handle(dst_handle, rb.target_width, rb.target_height, RK_FORMAT_RGB_888);
    int ret = imcheck(src_img, dst_img, {}, {});

    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    ret = imresize(src_img, dst_img);
    if (ret != IM_STATUS_SUCCESS) {
        printf("running RGA failed, %s\n", imStrError((IM_STATUS)ret));
        return -1;
    }

    if (src_handle > 0) releasebuffer_handle(src_handle);
    if (dst_handle > 0) releasebuffer_handle(dst_handle);
    
    return 0;

    // // init rga context
    // rga_buffer_t src;
    // rga_buffer_t dst;
    // im_rect      src_rect;
    // im_rect      dst_rect;
    // memset(&src_rect, 0, sizeof(src_rect));
    // memset(&dst_rect, 0, sizeof(dst_rect));
    // memset(&src, 0, sizeof(src));
    // memset(&dst, 0, sizeof(dst));
    // src = wrapbuffer_virtualaddr(src_buf, rb.in_width, rb.in_height, RK_FORMAT_RGB_888);
    // dst = wrapbuffer_virtualaddr(dst_buf, rb.target_width, rb.target_height, RK_FORMAT_RGB_888);
    // int ret = imcheck(src, dst, src_rect, dst_rect);
    // if (IM_STATUS_NOERROR != ret)
    // {
    //     fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
    //     return -1;
    // }
    // IM_STATUS STATUS = imresize(src, dst);
    // return 0;
}

int cpu_resize(void *src_buf, void *dst_buf, RESIZE_BOX &rb)
{
    // Only support RGB--->RGB
    std::cout << rb.in_width << " " << rb.in_height << " " << rb.target_width << " " << rb.target_height << std::endl;
    int reti = 0;
    reti = crop_and_scale_image_c(3, (unsigned char*)src_buf, rb.in_width, rb.in_height,
            0, 0, rb.in_width, rb.in_height,
            (unsigned char*)dst_buf, rb.target_width, rb.target_height,
            0, 0, rb.target_width, rb.target_height);

    if (reti != 0) {
        printf("Convert image by cpu failed %d\n", reti);
        return -1;
    }
    return 0;
}

int _resize_(void *src_buf, void *dst_buf, RESIZE_BOX &rb){
    int ret = 0;
// #ifdef ENABLE_RGA
//     ret = rga_resize(src_buf, dst_buf, rb);
// #else
//     ret = -1;
// #endif
    ret = -1;
    if (ret != 0)
    {
        ret = cpu_resize(src_buf, dst_buf, rb);
        return ret;
    }
    return ret;
}

static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
    if (dst == NULL) {
        printf("dst buffer is null\n");
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
    //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
    // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
    //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

    // 从原图指定区域取数据，双线性缩放到目标指定区域
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;    // down
            if (src_y == src_height - 1) {
                // 如果到图像最下边缘，变成选择上面的像素
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;            // right
            int index4 = index2 + 1 * channel;            // down right
            if (src_x == src_width - 1) {
                // 如果到图像最右边缘，变成选择左边的像素
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
            //     dst_x, dst_y, dst_x_offset, dst_y_offset,
            //     src_x, src_y, x_diff, y_diff,
            //     index1, index2, index3, index4);

            for (int c = 0; c < channel; c++) {
                unsigned char A = src[index1+c];
                unsigned char B = src[index3+c];
                unsigned char C = src[index2+c];
                unsigned char D = src[index4+c];

                unsigned char pixel = (unsigned char)(
                    A * (1 - x_diff) * (1 - y_diff) +
                    B * x_diff * (1 - y_diff) +
                    C * y_diff * (1 - x_diff) +
                    D * x_diff * y_diff
                );

                dst[(dst_y * dst_width  + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}