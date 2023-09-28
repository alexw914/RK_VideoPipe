#include "player.h"
#include "include/timer.h"

void _frame_callback(void* userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void* data){
    
    TIMER timer;
    timer.indent_set("    ");    
    app_context_t* ctx = (app_context_t*)userdata;

    int ret = 0;
    void *resize_buf = nullptr;
    resize_buf = malloc(width * height * 3);
    memset(resize_buf, 0, width * height * 3);

    rga_buffer_t src;
    rga_buffer_t dst;
    src = wrapbuffer_virtualaddr((void*)data, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
    dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);
    imcopy(src, dst);
    // // for debug

    cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);

    // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    // cv::imwrite("resize_input.jpg", resize_img);

    timer.tik();
    ctx->rkyolov5->predict((void*)resize_buf);
    timer.tok();
    timer.print_time("predict");
    // timer.tik();
    // ctx->rkyolov5->draw_box(resize_img);
    // timer.tok();
    // timer.print_time("save picture");

    if (resize_buf) free(resize_buf);    

}