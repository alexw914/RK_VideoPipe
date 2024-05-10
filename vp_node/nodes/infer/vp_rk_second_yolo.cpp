#include "vp_rk_second_yolo.h"
#include "objects/vp_sub_target.h"

namespace vp_nodes {
    vp_rk_second_yolo::vp_rk_second_yolo(std::string node_name, 
                                        std::string config_path, 
                                        std::vector<int> p_class_ids_applied_to,
                                        int min_width_applied_to, int min_height_applied_to):
                                        vp_secondary_infer_node(node_name, "", "", "", 1, 1, 1, p_class_ids_applied_to, min_width_applied_to, min_height_applied_to) {
        YOLOConfig conf;
        int ret = YOLO::load_config(config_path, conf);
        rk_model = std::make_shared<YOLO>(conf);
        this->initialized();
    }
    
    vp_rk_second_yolo::~vp_rk_second_yolo() {
        deinitialized();
        rk_model.reset();
    }

    void vp_rk_second_yolo::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_secondary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // infer using trt_vehicle library
        start_time = std::chrono::system_clock::now();
        std::vector<std::vector<DetectionResult>> res_datas;
        rk_model->run(mats_to_infer, res_datas);

        auto &frame_meta = frame_meta_with_batch[0];
        auto index = 0;

        for (int i = 0; i < res_datas.size(); i++) {
            for (int j = index; j < res_datas[i].size(); j++) {

                auto res = res_datas[i][j];
                // check value range
                auto x = std::max(0, res.box.top + frame_meta->targets[i]->x - 10);  // offset
                auto y = std::max(0, res.box.left + frame_meta->targets[i]->y - 8);   // offset
                auto w = std::min(res.box.bottom - res.box.top, frame_meta->frame.cols - x);
                auto h = std::min(res.box.right - res.box.left, frame_meta->frame.rows - y);
                if (w <= 0 || h <=0) {
                    continue;
                }
                
                // create sub target and update back into frame meta
                // we treat vehicle plate as sub target of those in vp_frame_meta.targets
                auto sub_target = std::make_shared<vp_objects::vp_sub_target>(x, y, w, h, 
                                                    res.id, res.score, res.label, frame_meta->frame_index, frame_meta->channel_index);
                frame_meta->targets[i]->sub_targets.push_back(sub_target);
            }
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);
        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_rk_second_yolo::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}