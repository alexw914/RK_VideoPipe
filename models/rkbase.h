# pragma once
#include <iostream>
#include <vector>
#include "rknn_api.h"
#include "resize_function.h"

#include "config.h"
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class RKBASE
{
public:
    RKBASE(const std::string& model_path);
    ~RKBASE();
    int set_coremask(int n);

protected:
    int model_channel = 3;
    int model_width = 640;
    int model_height = 384;

    rknn_context ctx;
    unsigned char* model_data = nullptr;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    rknn_input inputs[1];
    int ret;

    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;
};

// class {
// private:
//     std::vector<std::string> labels;
//     std::vector<std::string> alarm_labels;
// public:
//     void draw_results(cv::Mat &src, cv::Mat& dst, const std::vector<DetectionResult> &res);
//     void results_to_json(const std::vector<DetectionResult> &res, json &json_res);
// };

// void YOLO::draw_results(cv::Mat &src)
// {
//     int idx;
//     if (detection_results.size() == 0) return;
//     for (auto &obj : detection_results)
//     {
//         idx = obj.label + 3;
//         putText(src, format("(%d) %.1f%%", obj.label, obj.prob * 100), Point(obj.rect.x, obj.rect.y - 5),
//                 0, 0.6, cv::Scalar(255, 0, 0), 1, LINE_AA);
//         cv::rectangle(src, obj.rect, cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255), 2);
//     }
// }

// void YOLO::result_to_json()
// {
//     json_results.clear();
//     if (detection_results.empty()) return;
//     for (auto &obj : detection_results){
//         std::vector<float> tlbr = {obj.rect.x, obj.rect.y, obj.rect.x+obj.rect.width, obj.rect.y+obj.rect.height};
//         json_results["classes"].push_back(post->get_label(obj.label));
//         json_results["boxes"].push_back(tlbr);
//         json_results["scores"].push_back(obj.prob);
//     }
// }
// void RTMPOSE::draw_results(cv::Mat &src)
// {
// 	if (keypoint_results.size() == 0) return;
// 	for (auto result : keypoint_results)
// 	{
// 		for (int k = 0; k < keypoint_num; k++) {
// 			std::pair<int, int> joint_links = coco_17_joint_links[k];
// 			cv::Scalar s;
// 			if (k <= 5) {
// 				s = cv::Scalar(169, 169, 169);
// 			}
// 			else if(k > 5 && k <=11){
// 				s = cv::Scalar(147, 20, 255);
// 			}
// 			else{
// 				s = cv::Scalar(139, 139, 0);
// 			}
//             cv::circle(src, cv::Point2d(result.keypoints[k][0], result.keypoints[k][1]), 3,  s , 2);
//             cv::line(src, cv::Point2d(result.keypoints[joint_links.first][0], result.keypoints[joint_links.first][1]),
//                     cv::Point2d(result.keypoints[joint_links.second][0], result.keypoints[joint_links.second][1]), s, 2);   
//         }
// 	}
// }