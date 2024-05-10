#pragma once

#include "vp_primary_infer_node.h"
#include "yolo.h"

namespace vp_nodes {
    // yolo detector based on rknn
    class vp_rk_first_yolo: public vp_primary_infer_node
    {
    private:
        std::shared_ptr<YOLO> rk_model;
    protected:
        // we need a totally new logic for the whole infer combinations
        // no separate step pre-defined needed in base class
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        // override pure virtual method, for compile pass
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_rk_first_yolo(std::string node_name, std::string json_path);
        ~vp_rk_first_yolo();
    };
}