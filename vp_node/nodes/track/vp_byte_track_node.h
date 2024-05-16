#pragma once

#include <vector>
#include <map>
#include "vp_track_node.h"
#include "BYTETracker.h"

namespace vp_nodes {
    // track node using bytetrack
    class vp_byte_track_node : public vp_track_node
    {
    private:
        // track for
        vp_track_for track_for = vp_track_for::NORMAL;
        std::map<int, BYTETracker> all_trackers;
        double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final;
        virtual void track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects,
                           const std::vector<std::vector<float>>& target_embeddings, std::vector<int>& track_ids) override { return; };

    public:
        vp_byte_track_node(std::string node_name, vp_track_for track_for=vp_track_for::NORMAL);
        virtual ~vp_byte_track_node();

    };

}

