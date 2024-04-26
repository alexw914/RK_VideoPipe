#include <iostream>
#include <string.h>
#include "timer.h"
#include "yolo.h"
#include "rtmpose.h"
#include "classifier.h"


static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec);} 

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

int main(int argc, char** argv) 
{
    TIMER timer;
    timer.indent_set("");
    YOLOConfig conf;
    conf.model_path = std::string("assets/rknns/person_yolov5.rknn");
    conf.labels.push_back("person");
    conf.alarm_labels.push_back("person");
    conf.type = ModelType::YOLOv5;

    PoseConfig p_conf;
    p_conf.model_path = std::string("assets/rknns/rtmpose.rknn");
    p_conf.type = ModelType::RTMPose;

    ClsConfig c_conf;
    c_conf.model_path = std::string("assets/rknns/sit_resnet18.rknn");
    c_conf.type = ModelType::MobileNetv3;
    c_conf.labels.push_back("sit");
    c_conf.labels.push_back("stand");
    c_conf.alarm_labels.push_back("sit");

    auto rtmpose = new RTMPose(p_conf);
    auto yolo = new YOLO(conf);
    auto cls = new Classifier(c_conf);

    cv::Mat src = cv::imread("assets/images/005183.jpg");
    cv::Mat crop = cv::imread("assets/images/crop.jpg");

    std::vector<DetectionResult> det_res;
    // std::vector<KeyPointResult>
    KeyPointResult kpt_res;
    ClsResult cls_res;
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    timer.tik();
    yolo->run(src, det_res);
    timer.tok();
    timer.print_time("YOLO opencv");

    timer.tik();
    cls->run(crop, cls_res);
    timer.tok();
    std::cout << "Cls Result, label: " << cls_res.label << " score: " << cls_res.score << std::endl;
    timer.print_time("Cls");

    timer.tik();
    rtmpose->run(crop, kpt_res);
    timer.tok();
    timer.print_time("RTMPose");

    cv::cvtColor(src, src, cv::COLOR_RGB2BGR);
    for (auto r : det_res)
    {
        cv::putText(src, cv::format("(%s) %.1f%%", r.label.c_str(), r.score * 100), cv::Point(r.box.top, r.box.left - 5),
                0, 0.6, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::rectangle(src, cv::Point(r.box.top, r.box.left), cv::Point(r.box.bottom, r.box.right), cv::Scalar(255, 0, 0), 2);
    }

    std::vector<std::pair<int, int>> coco_17_joint_links = {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};
    for (int k = 0; k < 17; k++) {
        std::pair<int, int> joint_links = coco_17_joint_links[k];
        cv::Scalar s;
        if (k <= 5) s = cv::Scalar(169, 169, 169);
        else if(k > 5 && k <=11) s = cv::Scalar(147, 20, 255);
        else s = cv::Scalar(139, 139, 0);
        cv::circle(crop, cv::Point2d(kpt_res.keypoints[k].first, kpt_res.keypoints[k].second), 3,  s , 2);
        cv::line(crop, cv::Point2d(kpt_res.keypoints[joint_links.first].first, kpt_res.keypoints[joint_links.first].second),
                cv::Point2d(kpt_res.keypoints[joint_links.second].first, kpt_res.keypoints[joint_links.second].second), s, 2);   
    }
    cv::imwrite("./out.jpg", src);
    cv::imwrite("./crop_out.jpg", crop);
    return 0;
}