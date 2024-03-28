#include <iostream>
#include <fstream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "yolo.h"
#include "mpp_decoder.h"
#include "mk_mediakit.h"
#include "timer.h"
#include "ThreadPool.h"

#define LOGD 
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
// ZLMediakit, 
typedef struct {
    int cam_idx;
    const char *rtsp_url = nullptr;
    int frame_width;
    int frame_height;
    int frame_rate;
    int frame_idx = 0;
    MppDecoder *decoder = nullptr;
    // YOLO *yolo = nullptr;            // Detection model
    // BYTETracker *tracker = nullptr;  // track
} app_context_t;

void API_CALL on_track_frame_out(void *user_data, mk_frame frame) {
    app_context_t *ctx = (app_context_t *)user_data;
    LOGD("on_track_frame_out ctx=%p\n", user_data);
    const char* data = mk_frame_get_data(frame);
    size_t size = mk_frame_get_data_size(frame);
    LOGD("data: %s, size: %d", data, size);
    LOGD("decoder=%p\n", ctx->decoder);
    ctx->decoder->Decode((uint8_t *)data, size, 0);
}

void API_CALL on_mk_play_event_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[],
                                    int track_count) {
    if (err_code == 0) {
        //success
        printf("Play success!\n");
        int i;
        printf("Track count : %d ", track_count);
        for (i = 0; i < track_count; ++i)
        {
            LOGD("Track i : %d \n", tracks[i]);
            if (mk_track_is_video(tracks[i]))
            {
                log_info("Got video track: %s", mk_track_codec_name(tracks[i]));
                //监听track数据回调
                mk_track_add_delegate(tracks[i], on_track_frame_out, user_data);
            }
        }
    } else {
        LOGD("play failed: %d %s", err_code, err_msg);
    }
}

void API_CALL on_mk_shutdown_func(void *user_data, int err_code, const char *err_msg, mk_track tracks[], int track_count) {
    printf("play interrupted: %d %s", err_code, err_msg);
}

void process_video_rtsp(void* userdata)
{
    mk_config config;
    memset(&config, 0, sizeof(mk_config));
    config.log_mask = LOG_CONSOLE;
    mk_env_init(&config);
    mk_player player = mk_player_create();
    mk_player_set_on_result(player, on_mk_play_event_func, userdata);
    mk_player_set_on_shutdown(player, on_mk_shutdown_func, userdata);
    app_context_t *ctx = (app_context_t *)userdata;
    std::cout << ctx->rtsp_url << std::endl;
    mk_player_play(player, ctx->rtsp_url);
    printf("Enter any key to exit\n");
    getchar();
    if (player) mk_player_release(player);
}

void mpp_decoder_callback(void* userdata, int width_stride, int height_stride, int width, int height, int format, int fd, void* data){
    app_context_t *ctx = (app_context_t *)userdata;
    int ret = 0;
    void *resize_buf = nullptr;
    resize_buf = malloc(width * height * 3);
    memset(resize_buf, 0, width * height * 3);
    if (ctx->frame_idx % 5 == 0){
        // yuv --> rgb, RGA
        rga_buffer_t src;
        rga_buffer_t dst;
        src = wrapbuffer_virtualaddr((void*)data, width, height, RK_FORMAT_YCbCr_420_SP, width_stride, height_stride);
        dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);
        imcopy(src, dst);
        // debug
        cv::Mat decode_img(cv::Size(width, height), CV_8UC3, resize_buf);
        // run model ...
    }
    std::cout << "camera index: " << ctx->cam_idx << " frame index: " << std::to_string(ctx->frame_idx) << std::endl;
    ctx->frame_idx++;
    if (resize_buf) free(resize_buf);
}
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

int main(int argc, char** argv)
{
    std::ifstream in("assets/camera.txt");
    std::string s;
    std::vector<std::string> rtsp_urls;

    while (getline(in, s)) {
        rtsp_urls.push_back(s);
    }

    int cam_num = rtsp_urls.size();


    std::thread decode_threads[cam_num];
    app_context_t app_ctx[cam_num];

    for (auto idx = 0; idx < cam_num; idx++)
    {
        if (app_ctx[idx].decoder == nullptr) {
            auto decoder = new MppDecoder();
            decoder->Init(265, 25, &app_ctx[idx]);
            decoder->SetCallback(mpp_decoder_callback);
            app_ctx[idx].decoder = decoder;
        }

        app_ctx[idx].cam_idx = idx;
        app_ctx[idx].rtsp_url = rtsp_urls[idx].data();

        decode_threads[idx] = std::thread(process_video_rtsp, &app_ctx[idx]);
    }

    for (auto& t: decode_threads) {
        t.join();
    }
    return 0;
}
