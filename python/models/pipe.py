from .classifier import Classifier
from .detector import Detector
import cv2
import numpy as np

def clamp(x, max_):
    if x < 0:
        return 0
    else:
        return min(x, max_)

class DetClsPipe():

    def __init__(self, det_conf: dict, cls_conf: dict) -> None:

        self.det = Detector(det_conf)
        self.cls = Classifier(cls_conf)
        self.labels = cls_conf["labels"]
        self.alarm_labels = cls_conf["alarm_labels"]

    def run(self, img, reg_pts=None):
        # img : numpy, read by opencv
        det_res = self.det.run(img, reg_pts)
        if not det_res:
            return det_res
        
        mix_res = []
        for res in det_res:
            if res["label"] != "person":
                continue
            x1, y1, x2, y2 = res["box"]
            x1 = clamp(x1, img.shape[1])
            x2 = clamp(x2, img.shape[1])
            y1 = clamp(y1, img.shape[0])
            y2 = clamp(y2, img.shape[0])
            cls_res = self.cls.run(img[y1:y2, x1:x2])
            new_res = {'box': res['box'], 'label': cls_res[0]['label'], 'score': cls_res[0]['score'] * res["score"]}
            if new_res["score"] >= 0.65:
                mix_res.append(new_res)
        return mix_res
    
    def draw(self, img, res:list):
        src = img.copy()
        if res == []:
            return src
        else:
            for obj in res:
                box   = obj["box"]
                label = obj["label"]
                score = obj["score"]
                s = [0, 0, 255] if label in self.cls.alarm_labels else [0, 255, 0]
                cv2.rectangle(src, (box[0], box[1]), (box[2], box[3]), s, 2)
                cv2.putText(src, '({0}) {1:.1f}%'.format(label, score*100),
                            (box[0], box[1] - 6), cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (128, 0, 0), 2)
        return src
    

class DetDetPipe():
    def __init__(self, first_conf: dict, second_conf: dict) -> None:

        self.det_first  = Detector(first_conf)
        self.det_second = Detector(second_conf)
        self.labels       = second_conf["labels"]
        self.alarm_labels = second_conf["alarm_labels"]

    def mask_roi(self, img, det_res):

        roi_mask = np.zeros(img.shape, dtype=np.uint8)
        #创建mask层
        for res in det_res:
            if res["label"] != "person":
                continue    
            x1, y1, x2, y2 = res["box"]
            x1 = clamp(x1, img.shape[1])
            x2 = clamp(x2, img.shape[1])
            y1 = clamp(y1, img.shape[0])
            y2 = clamp(y2, img.shape[0])
            cv2.rectangle(roi_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, roi_mask)
        return masked_img  
     
    def run(self, img, reg_pts=None):
        # img : numpy, read by opencv
        first_res = self.det_first.run(img, reg_pts)
        if not first_res:
            return first_res
        # masked_img = self.mask_roi(img, first_res)
        # second_res = self.det_second.run(masked_img, reg_pts)
        final_res = []
        for res in first_res:
            if res["label"] != "person":
                continue
            x1, y1, x2, y2 = res["box"]
            x1 = clamp(x1, img.shape[1])
            x2 = clamp(x2, img.shape[1])
            y1 = clamp(y1, img.shape[0])
            y2 = clamp(y2, img.shape[0])
            second_res = self.det_second.run(img[y1:y2, x1:x2])
            max_res = None
            for r in second_res:
                if max_res is None:
                    max_res = r
                if max_res['score'] < r['score']:
                    max_res = r
            if max_res is not None:
                box = max_res['box']
                final_res.append({"box": [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                                   "label": max_res["label"], "score": max_res["score"]})
        return final_res
    
    def draw(self, img, det_res:list):
        return self.det_second.draw(img, det_res)
        

