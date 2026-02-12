#pragma once

#include <SDL2/SDL.h>
#include <string>
#include <atomic>
#include <chrono>

#include "nodes/base/vp_des_node.h"

namespace vp_nodes {
    // SDL2 hardware accelerated destination node
    // Renders NV12 frames directly using SDL2 for maximum performance
    class vp_sdl2_des_node: public vp_des_node {
    private:
        bool osd;
        bool show_fps;
        bool enable_vsync;
        std::string sdl_video_driver;
        std::string sdl_render_driver;

        // SDL2 components
        SDL_Window* window = nullptr;
        SDL_Renderer* renderer = nullptr;
        SDL_Texture* texture = nullptr;
        bool sdl_inited = false;

        // Display size
        int display_width = 0;
        int display_height = 0;

        // FPS tracking
        uint64_t fps_start_us = 0;
        uint32_t frame_count = 0;
        uint64_t last_log_time_us = 0;
        uint32_t last_log_frames = 0;

        // Get current time in microseconds
        static uint64_t now_us();

        // Draw FPS overlay on screen
        void draw_fps_overlay();

        // Initialize SDL2 for given frame size
        int init_sdl(int width, int height);

        // Render NV12 frame
        int render_frame_nv12(void* nv12_data, int stride_h, int width, int height);

        // Cleanup SDL2 resources
        void cleanup_sdl();

        // Handle SDL events (quit, escape)
        void handle_sdl_events();

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta>
        handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta>
        handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

    public:
        vp_sdl2_des_node(std::string node_name,
                         int channel_index,
                         bool osd = true,
                         bool show_fps = true,
                         bool enable_vsync = false,
                         const std::string& sdl_video_driver = "",
                         const std::string& sdl_render_driver = "");
        ~vp_sdl2_des_node();

        virtual std::string to_string() override;
    };
}
