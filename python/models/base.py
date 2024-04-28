import cv2
import numpy as np
from random import *
from rknnlite.api import RKNNLite
from shapely.geometry import Point, Polygon

class LetterBox:

    def __init__(self, new_shape=(384, 640), color=(114,114,114)) -> None:
        self.new_shape = new_shape
        self.color = color

    def __call__(self, img):
        
        # 每次返回dw, dh防止多线程时，处理不同分辨率造成不同步, 坐标还原错误
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        ratio = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        
        # Compute padding
        # ratio = r , r  # width, height ratios
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        scale = 1 / ratio
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        return img, (dw, dh, scale)

    def reverse(self, boxes, factor):
        if boxes is None:
            return boxes
        box_rev = np.copy(boxes)
        dw, dh, scale = factor
        box_rev[:, 0] = (box_rev[:, 0] - dw) * scale  # top left x
        box_rev[:, 1] = (box_rev[:, 1] - dh) * scale
        box_rev[:, 2] = (box_rev[:, 2] - dw) * scale       
        box_rev[:, 3] = (box_rev[:, 3] - dh) * scale
        return box_rev
    
class RKBase:

    def  __init__(self, model_path) -> None:
        
        self.model_path = model_path
        self.rknn = RKNNLite()
        self.init_env()
        
    def init_env(self):

        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print("Load RKNN rknnModel failed")
            exit(ret)
        id = randint(0, 2)
        if id == 0:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        elif id == 1:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        elif id == 2:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)

        if ret != 0:
            print("Init runtime environment failed")
            raise RuntimeError

class DetectorBase:

    def __init__(self, config) -> None:

        self.labels = config["labels"]
        self.alarm_labels = config["alarm_labels"]
        self.colors = config["colors"]
    
    def draw(self, img, det_res:list):
        """
        img numpy.array (read from opencv)
        det_res: list
        [{'box': [605, 113, 813, 358], 'label': 'person', 'score': 0.9450981020927429}]
        """
        src = img.copy()
        if det_res == []:
            return src
        else:
            for obj in det_res:
                box   = obj["box"]
                label = obj["label"]
                score = obj["score"]
                cv2.rectangle(src, (box[0], box[1]), (box[2], box[3]), self.colors[self.labels.index(label)], 2)
                cv2.putText(src, '({0}) {1:.1f}%'.format(label, score*100),
                            (box[0], box[1] - 6), cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (128, 0, 0), 2)
        return src
    
    def area_select(self, det_res:list, reg_pts=None):
        '''
        params: 
            det_res: get from parse_result function
            reg_pts: points of region (number of points must more than 2)
        '''
        new_res = []
        det_region = Polygon(reg_pts)
        for obj in det_res:
            center = Point((obj['box'][0] + obj['box'][2]) / 2, (obj['box'][1] + obj['box'][3]) / 2)
            if det_region.contains(center):
                new_res.append(obj)
        return new_res 

    def parse_result(self, results, reg_pts=None):
        '''
        params:
            results: Get from YOLOv5 result include (boxes, classes, scores)
            reg_pts 
        return:
            list: []
            [{'box': [605, 113, 813, 358], 'label': 'person', 'score': 0.9450981020927429}]
        '''
        boxes, classes, scores = results
        det_res = []
        if boxes is None:
            return det_res
        else:
            for box, cl, score in zip(boxes, classes, scores):
                obj = {'box': box.astype(int).tolist(), 'label': self.labels[cl], 'score': float(round(score, 4))}
                det_res.append(obj)
            if reg_pts is not None and len(reg_pts) >= 3:
                det_res = self.area_select(det_res, reg_pts)
        return det_res
    