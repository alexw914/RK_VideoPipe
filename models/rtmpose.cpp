#include "rtmpose.h"
#include <fstream>

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

int RTMPose::load_config(const std::string &json_path, PoseConfig &conf)
{
    std::ifstream f(json_path);
    json j_conf;
    f >> j_conf;
    conf.model_path = j_conf["model_path"].template get<std::string>();
    conf.keypoint_num = j_conf["keypoint_num"].template get<int>();
    conf.type = ModelType::RTMPose;
    return 0;	
}

void RTMPose::run(const cv::Mat &src, KeyPointResult &res)
{
    LETTER_BOX lb;
    init_letterbox(lb);
    if (src.cols == model_width && src.rows == model_height)
    {
        run_model(src.data, lb, res);
    }
    else{
        set_letterbox(src.cols, src.rows, lb);
        lb.reverse_available = true;
        cv::Mat lb_img;
        opencv_letter_box_resize(src, lb_img, lb);
        cv::cvtColor(lb_img, lb_img, cv::COLOR_BGR2RGB);
        run_model(lb_img.data, lb, res);
    }
}

void RTMPose::run(std::vector<cv::Mat> &img_datas, std::vector<KeyPointResult> &res_datas)
{
    res_datas.clear();
    // scan 1 by 1
    for (int i = 0; i < img_datas.size(); i++) {
        KeyPointResult res;
        this->run(img_datas[i], res);
        res_datas.push_back(res);
    }
}


void RTMPose::run_model(void* buf, LETTER_BOX &lb, KeyPointResult &result)
{
	inputs[0].buf = buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].index = i;
		outputs[i].want_float = 1;
	}
	ret = rknn_run(ctx, NULL);
	ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
	int e_width = output_attrs[0].dims[2];
	int e_height = output_attrs[1].dims[2];
	float* x_result = (float*)outputs[0].buf;
	float* y_result = (float*)outputs[1].buf;
	postprocess(x_result, y_result, e_width, e_height, lb, result);
	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
}

int RTMPose::postprocess(float *simcc_x_result, float *simcc_y_result, int extend_width, int extend_height, LETTER_BOX &lb, KeyPointResult& keypoint_result)
{
	for (int i = 0; i < config.keypoint_num; ++i)
	{
		// find the maximum and maximum indexes in the value of each Extend_width length
		auto x_biggest_iter = std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
		int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
		int pose_x = max_x_pos / 2;
		float score_x = *x_biggest_iter;

		// find the maximum and maximum indexes in the value of each exten_height length
		auto y_biggest_iter = std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
		int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
		int pose_y = max_y_pos / 2;
		float score_y = *y_biggest_iter;

		//float score = (score_x + score_y) / 2;
		float score = std::max(score_x, score_y);
		
		keypoint_result.keypoints.push_back(std::make_pair(w_reverse(pose_x, lb), h_reverse(pose_y, lb)));
        keypoint_result.scores.push_back(score);
	}
	return 0;
}

void RTMPose::set_letterbox(int in_w, int in_h, LETTER_BOX& lb)
{
    lb.in_width = in_w;
    lb.in_height = in_h;
    lb.target_width = model_width;
    lb.target_height = model_height;
    compute_letter_box(&lb);
}

void RTMPose::init_letterbox(LETTER_BOX &lb)
{
    memset(&lb, 0, sizeof(LETTER_BOX));
    lb.channel = 3;
    lb.target_width = model_width;
    lb.target_height = model_height;
}