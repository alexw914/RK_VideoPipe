# coding: utf-8
import cv2
import numpy as np
from .base import *

class Classifier(RKBase):
    def __init__(self, config) -> None:

        RKBase.__init__(self, config["model_path"])
        self.labels       = config["labels"]
        self.alarm_labels = config["alarm_labels"]
        self.input_shape  = config["input_shape"]
        self.txt          = "Class Result: "

        self.txt_thickness = 2
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)
        
    def run(self, img, reg_pts=None):

        src = img.copy()
        # bgr --> rgb
        src = cv2.resize(src, self.input_shape, interpolation=cv2.INTER_LINEAR)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)   ## 分类模型的输入通道为RGB，需要进行转换
        x = self.rknn.inference([np.expand_dims(src, 0)])
        results = np.exp(x[0])/np.sum(np.exp(x[0]))
        scores = np.squeeze(results)
        cls_res = self.parse_result(scores)
        
        return cls_res
        
    def parse_result(self, scores, max_return=True):

        """
        scores: np.array (get from Classifier.run function)
        max_return: only return label with max score
        """
        a = np.argsort(scores)[::-1]
        # numbers of labels must equal to output scores
        assert (len(a) == len(self.labels))

        cls_res = []
        # return all scores
        for i in a:
            ans = {"label": self.labels[i], "score": float(round(scores[i], 4))}
            cls_res.append(ans)
            if max_return:
                break
        return cls_res

    def draw(self, img, res):
        # 可以添加水印
        if res == []:
            return img
        src = img.copy()
        tl = self.txt_thickness or round(0.002 * (src.shape[0] + src.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        # Get text size for in_count and out_count
        show_txt = self.txt + ": (" + str(res[0]['label']) + ") " + str(round(res[0]['score'] * 100, 3)) + "%"
        t_size_in = cv2.getTextSize(show_txt, 0, fontScale=tl / 2, thickness=tf)[0]
        # Calculate positions for counts label
        text_width = t_size_in[0]
        text_x = (src.shape[1] - text_width) // 2  # Center x-coordinate
        text_y = t_size_in[1]

        # Create a rounded rectangle for in_count
        cv2.rectangle(
            src, (text_x - 5, text_y - 5), (text_x + text_width + 7, text_y + t_size_in[1] + 7), self.count_color, -1
        )
        cv2.putText(
            src, str(show_txt), (text_x, text_y + t_size_in[1]), 0, tl / 2, self.count_txt_color, tf, lineType=cv2.LINE_AA
        )
        return src
    