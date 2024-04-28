import cv2
from trackers import BYTETracker
from counters import ObjectCounter
from models import YOLOv5
from loguru import logger


class TimeCounter:

    def __init__(self) -> None:

        # # 一个实例类只能一个时间段对应一个rtsp_url, 若多个时间段开启需要清除之前的缓存
        # self.model = YOLOv5(config["model_path"], config["labels"], 
        #                     config["conf_thresh"], config["nms_thresh"])
        # self.tracker = BYTETracker()
        # self.counter = ObjectCounter()
        # self.counter.set_args(classes_names=config["labels"], reg_pts=config["reg_pts"])
        # self.rtsp_url = None
        self.time_list = []

    # 如果有多个时间段，需要合并。后期添加功能
    def add_time(self, loop):

        if self.time_list == []:
            self.time_list.append(loop)
            return 
        else:
            # 左边界排序进行插入
            i = 0
            while i < len(self.time_list):
                if self.time_list[i][0] < loop[0]:
                    i = i + 1
                else:
                    self.time_list.insert(i, loop)
                    break
            self.merge_time()
    
    def merge_time(self):

        new_list = [self.time_list[0]]
        for k in range(1, len(self.time_list)):
            if new_list[-1][1] > self.time_list[k][0]:
                if self.time_list[k][1] > new_list[-1][1]:
                    new_list[-1][1] = self.time_list[k][1]
            else:
                new_list.append(self.time_list[k])
        self.time_list = new_list
            
    def reset(self):
        
        self.tracker.reset()
        self.counter.reset()

    def count(self, frame):

        results = self.model.detect(frame)
        tracks  = self.tracker.update(results, frame)
        im0     = self.counter.start_counting(frame, tracks)
        res = [{"in_count": self.models["counter"].in_counts,
                "out_count": self.models["counter"].out_counts,
                "now_count": len(results[0])}]
        return res, im0


if __name__ == "__main__":

    ct = TimeCounter()
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18], [1, 22]] 
    for interval in intervals:
        ct.add_time(interval)
    print(ct.time_list)
        
