#include "vp_byte_track_node.h"

namespace vp_nodes {

	vp_byte_track_node::vp_byte_track_node(std::string node_name,
		vp_track_for track_for) :
		vp_track_node(node_name, track_for) {
		this->initialized();
	}

	vp_byte_track_node::~vp_byte_track_node() {
		deinitialized();
	}

	double vp_byte_track_node::GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
	{
		float in = (bb_test & bb_gt).area();
		float un = bb_test.area() + bb_gt.area() - in;
		if (un < DBL_EPSILON)
			return 0;
		return (double)(in / un);
	}

	std::shared_ptr<vp_objects::vp_meta> vp_byte_track_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta)
	{
		// channel_index can be different each call
		auto channel_index = meta->channel_index;

		std::vector<DetectionResult> det_res;
		// data used for tracking
		std::vector<vp_objects::vp_rect> rects;      // rects of targets
		std::vector<int> track_ids;

		if (track_for == vp_track_for::NORMAL) {
			for (auto& i : meta->targets) {
				DetectionResult res;
				res.box = BOX{ i->get_rect().x, i->get_rect().y, i->get_rect().x + i->get_rect().width, i->get_rect().y + i->get_rect().height };
				res.id = i->primary_class_id;
				res.score = i->primary_score;
				res.label = i->primary_label;
				det_res.push_back(res);
				rects.push_back(i->get_rect());      // rect fo target (via i variable)
			}
		}

		if (track_for == vp_track_for::FACE) {
			for (auto& i : meta->face_targets) {
				DetectionResult res;
				res.box = BOX{ i->get_rect().x, i->get_rect().y, i->get_rect().x + i->get_rect().width, i->get_rect().y + i->get_rect().height };
				res.id = 0;
				res.score = i->score;
				res.label = std::string("label");
				det_res.push_back(res);
				rects.push_back(i->get_rect());      // rect fo target (via i variable)
			}
		}

		track_ids.resize(det_res.size());
		for (auto& it : track_ids) it = -1;
		
		// check if trackers are initialized or not for specific channel
		if (all_trackers.count(channel_index) == 0) {
			all_trackers[channel_index] = BYTETracker(25, 30);
			VP_INFO(vp_utils::string_format("[%s] initialize bytetracker the first time for channel %d", node_name.c_str(), channel_index));
		}
		auto& tracker = all_trackers[channel_index];

		std::vector<STrack> output_stracks = tracker.update(det_res);
		
		// update
		for (auto& it : output_stracks) {
			for (int i = 0; i < rects.size(); ++i) {
				if (GetIOU(cv::Rect_<float>(rects[i].x, rects[i].y, rects[i].width, rects[i].height),
					cv::Rect_<float>(it.tlwh[0], it.tlwh[1], it.tlwh[2], it.tlwh[3])) > 0.8) {
					track_ids[i] = it.track_id;
					//meta->targets[i]->x = it.tlwh[0];
					//meta->targets[i]->y = it.tlwh[1];
					//meta->targets[i]->width = it.tlwh[2];
					//meta->targets[i]->height = it.tlwh[3];
				}
			}
		}
		postprocess(meta, rects, std::vector<std::vector<float>>(), track_ids);

		return meta;
	}
}