import cv2, os, time, json
import numpy as np
from .base import *
    
class YOLOv5Post:

    def __init__(self, input_shape=(640, 384), conf_thresh=0.25, nms_thresh=0.45, anchors=None) -> None:
        
        self.masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]] if anchors is None else anchors
        self.conf_thresh = 0.25
        self.nms_thresh = 0.45
        self.shape = input_shape

    def set_args(self, conf_thresh, nms_thresh):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def xywh2xyxy(self, x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def process(self, input, mask):

        anchors = [self.anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])
        
        box_confidence = np.expand_dims(input[..., 4], axis=-1)

        box_class_probs = input[..., 5:]

        box_xy = input[..., :2]*2 - 0.5
        
        col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w).reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w).reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid

        box_xy *= int(self.shape[1]/grid_h)
        # box_xy *= input.shape[0]

        box_wh = pow(input[..., 2:4]*2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

        _box_pos = np.where(box_confidences >= self.conf_thresh)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.conf_thresh)

        return boxes[_class_pos], classes[_class_pos], (class_max_score * box_confidences)[_class_pos]


    def nms(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)    

    def __call__(self, rknn_out):

        boxes, classes, scores = [], [], []
        outputs = [np.transpose(o_.reshape([3, -1]+list(o_[0].shape[-2:])), (2, 3, 0, 1)) for o_ in rknn_out]
        for input, mask in zip(outputs, self.masks):
            b, c, s = self.process(input, mask)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = self.xywh2xyxy(np.concatenate(boxes))
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


class YOLOv5(RKBase):

    def __init__(self, model_path, labels:list, conf_thresh=0.25, nms_thresh=0.45, anchors=None, input_shape=(640, 384)) -> None:

        RKBase.__init__(self, model_path)
        self.labels = labels
        self.input_shape = input_shape
        self.lb = LetterBox(new_shape=(input_shape[1], input_shape[0]))
        self.post = YOLOv5Post(input_shape, conf_thresh, nms_thresh, anchors)
        self.post.set_args(conf_thresh, nms_thresh)

    def detect(self, img):

        # img : numpy.array (read img by opencv)
        src, factor = self.lb.__call__(img.copy())
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        rknn_out = self.rknn.inference([np.expand_dims(src, 0)])
        boxes, classes, scores = self.post.__call__(rknn_out)
        boxes = self.lb.reverse(boxes, factor)
        
        return boxes, classes, scores

