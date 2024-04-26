#include "vp_rk_second_rtmpose.h"

namespace vp_nodes {
    vp_rk_second_rtmpose::vp_rk_second_rtmpose(std::string node_name, 
                                        std::string config_path, 
                                        std::vector<int> p_class_ids_applied_to,
                                        int min_width_applied_to, int min_height_applied_to,
                                        vp_objects::vp_pose_type type):
                                        vp_secondary_infer_node(node_name, "", "", "", 1, 1, 1, p_class_ids_applied_to, min_width_applied_to, min_height_applied_to), type(type) {
        PoseConfig conf;
        int ret = RTMPose::load_config(config_path, conf);
        rk_model = std::make_shared<RTMPose>(conf);
        this->initialized();
    }
    
    vp_rk_second_rtmpose::~vp_rk_second_rtmpose() {
        rk_model.reset();
        deinitialized();
    }

    void vp_rk_second_rtmpose::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        std::vector<cv::Mat> mats_to_infer;

        // start
        auto start_time = std::chrono::system_clock::now();

        // prepare data, as same as base class
        vp_secondary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        // infer using trt_vehicle library
        start_time = std::chrono::system_clock::now();
        std::vector<KeyPointResult> res_datas;
        rk_model->run(mats_to_infer, res_datas);

        auto& frame_meta = frame_meta_with_batch[0];
        auto index = 0;
        for (int i = 0; i < res_datas.size(); i++) {
            for (int j = index; j < frame_meta->targets.size(); j++) {
                // need apply or not?
                if (!need_apply(frame_meta->targets[j]->primary_class_id, frame_meta->targets[j]->width, frame_meta->targets[j]->height)) {
                    // continue as its primary_class_id is not in p_class_ids_applied_to
                    continue;
                }

                std::vector<vp_objects::vp_pose_keypoint> kps;
                for (int k = 0; k < res_datas[i].keypoints.size(); k++){
                    auto x = std::max(0, res_datas[i].keypoints[k].first + frame_meta->targets[i]->x - 10);  // offset
                    auto y = std::max(0, res_datas[i].keypoints[k].second + frame_meta->targets[i]->y - 8);   // offset
                    kps.push_back(vp_objects::vp_pose_keypoint{i, x, y, res_datas[i].scores[k]});
                }
                auto pose_target = std::make_shared<vp_objects::vp_frame_pose_target>(type, kps);
                frame_meta->pose_targets.push_back(pose_target);

                // break as we found the right target!
                index = j + 1;
                break;
            }
        }
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);
        // can not calculate preprocess time and postprocess time, set 0 by default.
        vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
    }

    void vp_rk_second_rtmpose::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {

    }
}