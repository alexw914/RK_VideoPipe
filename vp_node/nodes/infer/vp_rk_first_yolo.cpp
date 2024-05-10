#include "vp_rk_first_yolo.h"

namespace vp_nodes {
    
    vp_rk_first_yolo::vp_rk_first_yolo(std::string node_name, std::string json_path):
                                                    vp_primary_infer_node(node_name, "") {
        YOLOConfig conf;
        int ret = YOLO::load_config(json_path, conf);
        rk_model = std::make_shared<YOLO>(conf);
        this->initialized();
    }
    
    vp_rk_first_yolo::~vp_rk_first_yolo() {
        deinitialized();
        // rk_model.reset();
    }

    // please refer to vp_infer_node::run_infer_combinations
    void vp_rk_first_yolo::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<DetectionResult>> res_datas;
        rk_model->run(mats_to_infer, res_datas);

        assert(res_datas.size() == 1);
        auto& res = res_datas[0];
        auto& frame_meta = frame_meta_with_batch[0];

        for (auto& obj : res){
            auto target = std::make_shared<vp_objects::vp_frame_target>(obj.box.top, obj.box.left, obj.box.bottom - obj.box.top, obj.box.right - obj.box.left, 
                                                                                    obj.id, obj.score, frame_meta->frame_index, frame_meta->channel_index, obj.label);            
            frame_meta->targets.push_back(target);
        }

        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_rk_first_yolo::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}