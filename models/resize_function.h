#ifndef _MZ_RESIZE_FUNCTION
#define _MZ_RESIZE_FUNCTION

#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef _MZ_LETTER_BOX
#define _MZ_LETTER_BOX
typedef struct _LETTER_BOX{
    int in_width=0, in_height=0;
    int target_width=0, target_height=0;
    int channel=3;

    float img_wh_ratio=1, target_wh_ratio=1;
    float resize_scale_w=0, resize_scale_h=0;
    int resize_width=0, resize_height=0;
    int h_pad_top=0, h_pad_bottom=0;
    int w_pad_left=0, w_pad_right=0;

    bool reverse_available=false;
} LETTER_BOX;
#endif

int compute_letter_box(LETTER_BOX* lb);
int print_letter_box_info(LETTER_BOX& lb);
int h_reverse(int h, LETTER_BOX& lb);
int w_reverse(int w, LETTER_BOX& lb);

int rga_letter_box_resize(int src_fd, int dst_fd, LETTER_BOX* lb);
int rga_letter_box_resize(void *src_buf, void *dst_buf, LETTER_BOX* lb);
int rga_letter_box_resize(void *src_buf, int dst_fd, LETTER_BOX* lb);

int cpu_letter_box_resize(void *src_buf, void *dst_buf, LETTER_BOX &lb, char pad_color=114);
int letter_box_resize(void *src_buf, void *dst_buf, LETTER_BOX &lb, char pad_color=114);

void opencv_letter_box_resize(const cv::Mat &image, cv::Mat &padded_image, LETTER_BOX& lb, const cv::Scalar& pad_color=cv::Scalar(114, 114, 114));

// cpu letterbox or resize, but it not as well as opencv
static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                  int crop_x, int crop_y, int crop_width, int crop_height,
                                  unsigned char *dst, int dst_width, int dst_height,
                                  int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height);

#ifndef _MZ_RESIZE_BOX
#define _MZ_RESIZE_BOX
typedef struct _RESIZE_BOX{
    int in_width=0, in_height=0;
    int target_width=0, target_height=0;
} RESIZE_BOX;
#endif
int rga_resize(void *src_buf, void *dst_buf, RESIZE_BOX &rb);
int cpu_resize(void *src_buf, void *dst_buf, RESIZE_BOX &rb);
int _resize_(void *src_buf, void *dst_buf, RESIZE_BOX &rb);

// int rga_convert_color(void* src_buf, void* dst_buf, int width, int height)
#endif