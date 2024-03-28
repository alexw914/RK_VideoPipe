#include "rtmpose.h"

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat &input_image, const DetectionResult &obj);
static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse=false);

void RTMPose::run(const cv::Mat &src, KeyPointDetectionResult& result)
{
	inputs[0].buf = src.data;
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
	postprocess(x_result, y_result, e_width, e_height, result);
	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
}

int RTMPose::postprocess(float *simcc_x_result, float *simcc_y_result, int extend_width, int extend_height, KeyPointDetectionResult& keypoint_result)
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

		std::array<float, 2> p;
		p[0] = pose_x;
		p[1] = pose_y;
		keypoint_result.keypoints.push_back(p);
        keypoint_result.scores.push_back(score);
	}
	return 0;
}

/*
PoseTracker YOLO + POSE + (ByteTrack)
Consider add ByteTrack
*/
void RTMPoseTracker::run(const cv::Mat &src, std::vector<DetectionResult>& det_res, std::vector<KeyPointDetectionResult>& kpt_res)
{
	yolo->run(src, det_res);
	for (auto &result : det_res)
	{
		auto result_pair = CropImageByDetectBox(src, result);
		cv::Mat crop_mat = result_pair.first;
		cv::Mat affine_transform_reverse = result_pair.second;
		// // BGR to RGB
		// cv::Mat input_mat_copy_rgb;
		KeyPointDetectionResult p_result;
		rtmpose->run(crop_mat, p_result);
		// anti affine transformation to obtain the coordinates on the original picture
		for (int i = 0; i < rtmpose->get_keypoint_num(); ++i)
		{
			cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
			origin_point_Mat.at<double>(0, 0) = p_result.keypoints[i][0];
			origin_point_Mat.at<double>(1, 0) = p_result.keypoints[i][1];
			cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;

			p_result.keypoints[i][0] = temp_result_mat.at<double>(0, 0);
			p_result.keypoints[i][1] = temp_result_mat.at<double>(1, 0);
		}
		kpt_res.emplace_back(p_result);
	}
}

// void RTMPoseTracker::run(image_buffer_t &src, std::vector<DetectionResult> &det_res, std::vector<KeyPointDetectionResult> &kpt_res)
// {
//     cv::Mat img(cv::Size(src.width, src.height), CV_8UC3, src.virt_addr);
// 	run(img, det_res, kpt_res);
// }

static std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat &input_image, const DetectionResult& obj)
{
	std::pair<cv::Mat, cv::Mat> result_pair;

	// deep copy
	cv::Mat input_mat_copy;
	input_image.copyTo(input_mat_copy);

	// calculate the width, height and center points of the human detection box
	int box_width    = obj.box.bottom - obj.box.top;
	int box_height   = obj.box.right - obj.box.left;
	int box_center_x = obj.box.top + box_width / 2;
	int box_center_y = obj.box.left + box_height / 2;
	float aspect_ratio = 192.0 / 256.0;

	// adjust the width and height ratio of the size of the picture in the RTMPOSE input
	if (box_width > (aspect_ratio * box_height))
	{
		box_height = box_width / aspect_ratio;
	}
	else if (box_width < (aspect_ratio * box_height))
	{
		box_width = box_height * aspect_ratio;
	}

	float scale_image_width = box_width * 1.2;
	float scale_image_height = box_height * 1.2;

	// get the affine matrix
	cv::Mat affine_transform = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256
	);

	cv::Mat affine_transform_reverse = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256,
		true
	);

	// affine transform
	cv::Mat affine_image;
	cv::warpAffine(input_mat_copy, affine_image, affine_transform, cv::Size(192, 256), cv::INTER_LINEAR);
	// cv::imwrite("affine_img.jpg", affine_image);

	result_pair = std::make_pair(affine_image, affine_transform_reverse);

	return result_pair;
}

static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse)
{
	// solve the affine transformation matrix

	// get the three points corresponding to the source picture and the target picture
	cv::Point2f src_point_1;
	src_point_1.x = center_x;
	src_point_1.y = center_y;

	cv::Point2f src_point_2;
	src_point_2.x = center_x;
	src_point_2.y = center_y - scale_width * 0.5;

	cv::Point2f src_point_3;
	src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
	src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


	float alphapose_image_center_x = output_image_width / 2;
	float alphapose_image_center_y = output_image_height / 2;

	cv::Point2f dst_point_1;
	dst_point_1.x = alphapose_image_center_x;
	dst_point_1.y = alphapose_image_center_y;

	cv::Point2f dst_point_2;
	dst_point_2.x = alphapose_image_center_x;
	dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

	cv::Point2f dst_point_3;
	dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
	dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


	cv::Point2f srcPoints[3];
	srcPoints[0] = src_point_1;
	srcPoints[1] = src_point_2;
	srcPoints[2] = src_point_3;

	cv::Point2f dstPoints[3];
	dstPoints[0] = dst_point_1;
	dstPoints[1] = dst_point_2;
	dstPoints[2] = dst_point_3;
	// get affine matrix
	cv::Mat affineTransform;
	if (inverse)
	{
		affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
	}
	else
	{
		affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
	}

	return affineTransform;
}