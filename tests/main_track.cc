#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include "yolo.h"
#include "BYTETracker.h"
#include <filesystem>

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

int main(int argc, char** argv) 
{
    std::filesystem::current_path("/root/.vs/RK_VideoPipe/41b606b4-6586-4527-af89-a26a0ab1539d/src");
    auto begin_time = std::chrono::steady_clock::now();
    double duration = .0;
    cv::VideoCapture cap;
    cap.open("assets/videos/vehicle.mp4");
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    spdlog::info("Total frames: {}, frame width: {}, frame height: {}.", nFrame, img_w, img_h);

    YOLOConfig conf;
    std::string json_path = "assets/configs/vehicle.json";
    int ret = YOLO::load_config(json_path, conf);
    auto yolo = std::make_shared<YOLO>(conf);
    
    cv::VideoWriter writer;
    writer = cv::VideoWriter("output.avi", cv::VideoWriter::fourcc('M','P','E','G'), fps, cv::Size(img_w, img_h));
    cv::Mat img;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 0;

    while (true)
    {
        if(!cap.read(img))
        {
            spdlog::info(" Break at cap.read(img) !");
            break;
        }
        num_frames ++;
        if (num_frames % 20 == 0)
            spdlog::info("Processing frame: {} Now FPS: {}", num_frames, num_frames * 1000 / total_ms);
        if (img.empty())
            break;

        auto start_time = std::chrono::steady_clock::now();
        std::vector<DetectionResult> det_res;
        yolo->run(img, det_res);

        std::vector<STrack> output_stracks = tracker.update(det_res);

        auto stop_time = std::chrono::steady_clock::now();
        duration = std::chrono::duration<double, std::milli>(stop_time - start_time).count();
        spdlog::info("Frame index: {}, model run use {} ms.", num_frames, duration);

        total_ms += duration;

        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
            cv::Point obj_center(tlwh[0] + tlwh[1] / 2, tlwh[1] + tlwh[3] / 2);
            cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
            putText(img, format("(%s: %d) %.2f", output_stracks[i].label.c_str(), output_stracks[i].track_id, output_stracks[i].score), Point(tlwh[0], tlwh[1] - 5),
                    0, 0.6, cv::Scalar(255, 0, 0), 2, LINE_AA);                
            rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }
        putText(img, format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000 / total_ms  , output_stracks.size()), 
                Point(0, 30), 0, 0.6, cv::Scalar(0, 255, 0), 2, LINE_AA);
        writer.write(img);
    }

    cap.release();
    auto end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>(end_time - begin_time).count();
    spdlog::info("All time: {} s. Total FPS: {}", duration, num_frames * 1000 / total_ms );
    return 0;
}
