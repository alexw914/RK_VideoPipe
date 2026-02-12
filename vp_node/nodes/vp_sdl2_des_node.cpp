#include <cstring>
#include <iostream>

#include "vp_utils/logger/vp_logger.h"
#include "vp_utils/vp_utils.h"
#include "vp_sdl2_des_node.h"

namespace vp_nodes {

uint64_t vp_sdl2_des_node::now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

vp_sdl2_des_node::vp_sdl2_des_node(std::string node_name,
                                     int channel_index,
                                     bool osd,
                                     bool show_fps,
                                     bool enable_vsync,
                                     const std::string& sdl_video_driver,
                                     const std::string& sdl_render_driver)
    : vp_des_node(node_name, channel_index),
      osd(osd),
      show_fps(show_fps),
      enable_vsync(enable_vsync),
      sdl_video_driver(sdl_video_driver),
      sdl_render_driver(sdl_render_driver) {
    this->initialized();
    VP_INFO(vp_utils::string_format("[%s] SDL2 destination node created",
                               node_name.c_str()));
}

vp_sdl2_des_node::~vp_sdl2_des_node() {
    cleanup_sdl();
    deinitialized();
}

std::string vp_sdl2_des_node::to_string() {
    return "sdl2_des";
}

int vp_sdl2_des_node::init_sdl(int width, int height) {
    if (sdl_inited)
        return 0;

    // Set SDL hints for performance
    if (!sdl_video_driver.empty() && sdl_video_driver[0]) {
        if (setenv("SDL_VIDEODRIVER", sdl_video_driver.c_str(), 1))
            VP_WARN("set SDL_VIDEODRIVER failed");
    }
    if (!sdl_render_driver.empty() && sdl_render_driver[0]) {
        if (setenv("SDL_HINT_RENDER_DRIVER", sdl_render_driver.c_str(), 1))
            VP_WARN("set SDL_HINT_RENDER_DRIVER failed");
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    SDL_SetHint(SDL_HINT_RENDER_BATCHING, "1");
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, enable_vsync ? "1" : "0");
#ifdef SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR
    SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "1");
#endif
#ifdef SDL_HINT_VIDEO_X11_FORCE_EGL
    SDL_SetHint(SDL_HINT_VIDEO_X11_FORCE_EGL, "1");
#endif

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_Init failed: %s",
                                   node_name.c_str(), SDL_GetError()));
        return -1;
    }

    Uint32 win_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    window = SDL_CreateWindow(
        "RK_VideoPipe - MPP HW Decode + SDL2",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width,
        height,
        win_flags
    );
    if (!window) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_CreateWindow failed: %s",
                                   node_name.c_str(), SDL_GetError()));
        cleanup_sdl();
        return -1;
    }

    Uint32 rflags = SDL_RENDERER_ACCELERATED;
    if (enable_vsync)
        rflags |= SDL_RENDERER_PRESENTVSYNC;

    renderer = SDL_CreateRenderer(window, -1, rflags);
    if (!renderer) {
        VP_WARN(vp_utils::string_format("[%s] SDL_CreateRenderer accelerated failed: %s, fallback",
                                    node_name.c_str(), SDL_GetError()));
        renderer = SDL_CreateRenderer(window, -1, enable_vsync ? SDL_RENDERER_PRESENTVSYNC : 0);
        if (!renderer) {
            VP_ERROR(vp_utils::string_format("[%s] SDL_CreateRenderer fallback failed: %s",
                                       node_name.c_str(), SDL_GetError()));
            cleanup_sdl();
            return -1;
        }
    }

    SDL_RendererInfo rinfo;
    if (SDL_GetRendererInfo(renderer, &rinfo) == 0) {
        VP_INFO(vp_utils::string_format("[%s] SDL video=%s renderer=%s flags=0x%x vsync=%d",
                                   node_name.c_str(),
                                   SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "unknown",
                                   rinfo.name ? rinfo.name : "unknown",
                                   rinfo.flags, enable_vsync));
    }

    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_NV12,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height
    );
    if (!texture) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_CreateTexture(NV12) failed: %s",
                                   node_name.c_str(), SDL_GetError()));
        cleanup_sdl();
        return -1;
    }

    display_width = width;
    display_height = height;
    sdl_inited = true;

    return 0;
}

void vp_sdl2_des_node::cleanup_sdl() {
    if (texture) {
        SDL_DestroyTexture(texture);
        texture = nullptr;
    }
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    if (window) {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    if (sdl_inited) {
        SDL_Quit();
        sdl_inited = false;
    }
}

void vp_sdl2_des_node::handle_sdl_events() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT)
            alive = false;  // This will stop all nodes in pipeline
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
            alive = false;
    }
}

int vp_sdl2_des_node::render_frame_nv12(void* nv12_data, int stride_h,
                                        int width, int height) {
    if (!nv12_data || !sdl_inited)
        return -1;

    const uint8_t* y_plane = (const uint8_t*)nv12_data;
    const uint8_t* uv_plane = y_plane + (size_t)stride_h * (size_t)height;

    void* pixels = nullptr;
    int pitch = 0;
    if (SDL_LockTexture(texture, nullptr, &pixels, &pitch) != 0) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_LockTexture failed: %s",
                                   node_name.c_str(), SDL_GetError()));
        return -1;
    }

    if (!pixels || pitch <= 0) {
        SDL_UnlockTexture(texture);
        return -1;
    }

    // Copy Y plane
    uint8_t* dst_y = (uint8_t*)pixels;
    for (int i = 0; i < height; ++i) {
        memcpy(dst_y + (size_t)i * (size_t)pitch,
               y_plane + (size_t)i * (size_t)stride_h,
               (size_t)width);
    }

    // Copy UV plane
    uint8_t* dst_uv = dst_y + (size_t)pitch * (size_t)height;
    for (int i = 0; i < height / 2; ++i) {
        memcpy(dst_uv + (size_t)i * (size_t)pitch,
               uv_plane + (size_t)i * (size_t)stride_h,
               (size_t)width);
    }

    SDL_UnlockTexture(texture);

    if (SDL_RenderCopy(renderer, texture, nullptr, nullptr) != 0) {
        VP_ERROR(vp_utils::string_format("[%s] SDL_RenderCopy failed: %s",
                                   node_name.c_str(), SDL_GetError()));
        return -1;
    }

    if (show_fps)
        draw_fps_overlay();

    SDL_RenderPresent(renderer);

    return 0;
}

void vp_sdl2_des_node::draw_fps_overlay() {
    if (frame_count == 0 || fps_start_us == 0)
        return;

    uint64_t elapsed = now_us() - fps_start_us;
    if (!elapsed)
        return;

    double fps = (double)frame_count * 1000000.0 / (double)elapsed;

    // Simple FPS text using SDL drawing primitives
    char fps_text[32];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);

    // Draw semi-transparent background
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
    SDL_Rect bg_rect = {10, 10, 120, 30};
    SDL_RenderFillRect(renderer, &bg_rect);

    // Note: For proper text rendering, we'd need SDL_ttf, but let's keep it simple
    // Just draw the FPS value as colored indicators for now
    // Green rectangle height indicates FPS (higher = better)
    int bar_height = (int)(fps / 3);  // Scale: 180fps = 60px
    if (bar_height > 50) bar_height = 50;
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    SDL_Rect fps_bar = {15, 50 - bar_height, 10, bar_height};
    SDL_RenderFillRect(renderer, &fps_bar);

    // Log to console periodically
    uint64_t now = now_us();
    if (!last_log_time_us) {
        last_log_time_us = now;
        last_log_frames = frame_count;
        return;
    }

    uint64_t delta_us = now - last_log_time_us;
    if (delta_us >= 1000000) {  // Log every second
        uint32_t delta_frames = frame_count - last_log_frames;
        double cur_fps = delta_us ? ((double)delta_frames * 1000000.0 / (double)delta_us) : 0.0;
        double avg_fps = elapsed ? ((double)frame_count * 1000000.0 / (double)elapsed) : 0.0;

        VP_INFO(vp_utils::string_format("[%s] FPS: cur=%.1f avg=%.1f frames=%u",
                                   node_name.c_str(), cur_fps, avg_fps, frame_count));

        last_log_time_us = now;
        last_log_frames = frame_count;
    }
}

std::shared_ptr<vp_objects::vp_meta>
vp_sdl2_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
    VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d",
                                  node_name.c_str(), meta->channel_index, meta->frame_index));

    // Initialize SDL on first frame
    if (!sdl_inited) {
        int w = meta->original_width;
        int h = meta->original_height;
        if (init_sdl(w, h) < 0) {
            VP_ERROR("Failed to initialize SDL2");
            return nullptr;
        }
    }

    // Start FPS tracking on first frame
    if (fps_start_us == 0) {
        fps_start_us = now_us();
    }

    frame_count++;

    // Handle NV12 format frame
    if (meta->is_nv12) {
        // Direct MPP decoded frame rendering
        render_frame_nv12(meta->nv12_data, meta->stride_h,
                         meta->original_width, meta->original_height);
    } else {
        // Fallback to BGR format (compatibility with existing pipeline)
        cv::Mat frame_to_show;
        if (osd && !meta->osd_frame.empty()) {
            frame_to_show = meta->osd_frame;
        } else if (!meta->frame.empty()) {
            frame_to_show = meta->frame;
        } else {
            return nullptr;
        }

        // Resize if needed
        if (display_width != frame_to_show.cols ||
            display_height != frame_to_show.rows) {
            cv::Mat resized;
            cv::resize(frame_to_show, resized,
                     cv::Size(display_width, display_height));
            frame_to_show = resized;
        }

        // Convert BGR to RGB and update texture
        cv::Mat rgb_frame;
        cv::cvtColor(frame_to_show, rgb_frame, cv::COLOR_BGR2RGB);

        SDL_UpdateTexture(texture, nullptr, rgb_frame.data, rgb_frame.step);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);

        if (show_fps)
            draw_fps_overlay();

        SDL_RenderPresent(renderer);
    }

    // Handle SDL events
    handle_sdl_events();

    return nullptr;  // Des node returns nullptr
}

std::shared_ptr<vp_objects::vp_meta>
vp_sdl2_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
    return nullptr;
}

} // namespace vp_nodes
