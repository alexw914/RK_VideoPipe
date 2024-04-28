import requests, json, os
from .yolov5 import YOLOv5
from .base import DetectorBase

class Detector(DetectorBase):

    def __init__(self, config: dict) -> None:
        super(Detector, self).__init__(config)
        self.model = YOLOv5(config["model_path"], config["labels"], 
                            config["conf_thresh"], config["nms_thresh"], config["anchors"])

    def run(self, img, reg_pts=None, parse_result=True):
        # img : numpy, read by opencv
        results = self.model.detect(img)
        if not parse_result:
            return results
        det_res = self.parse_result(results, reg_pts)
        return det_res


# class HttpDetector:
    
#     # 使用http接口, 推理
#     def __init__(self, port:int=5000) -> None:
#         self.burl  = 'http://localhost:' + str(port)
#         self.sess = requests.session()
#         self.model_map = {
#             "RYDD":  'fall',
#             "PLJS":  'pilaojiashi',
#             "WCFGY": 'people_vest',
#             "WCGZF": 'people_gongzuofu',
#             "WSJ":   'people_phone',
#             "RYJZ":  'people_jiuzuo',
#             "DJ":    'fight'
#         }
#         self.model_configs = {
#             "RYDD":  {"labels": ["fall"], "alarm_labels": ["fall"], "colors": [[0, 0, 255]]},
#             "PLJS":  {"labels": ["head", "open_eye", "closed_mouth", "open_mouth", "closed_eye"], "alarm_labels": ["open_mouth", "closed_eye"], "colors": [[0, 255, 0], [0, 255, 0], [0, 255, 0],[0, 0, 255],[0, 0, 255]]},
#             "WCFGY": {"labels": ["no_vest", "vest"], "alarm_labels": ["no_vest"], "colors": [[0, 0, 255], [0, 255, 0]]},
#             "WCGZF": {"labels": ['no working_clothes', 'working_clothes'], "alarm_labels": ['no working_clothes'], "colors": [[0, 0, 255], [0, 255, 0]]},
#             "WSJ":   {"labels": ["phone"], "alarm_labels": ["phone"], "colors": [[0, 0, 255]]},
#             "RYJZ":  {"labels": ['jiuzuo','zhanli'], "alarm_labels": ['jiuzuo'], "colors": [[0, 0, 255], [0, 255, 0]]},
#             "DJ":    {"labels": ['fight'], "alarm_labels": ['fight'], "colors": [[0, 0, 255]]},             
#         }
#         self.model_base = {}
#         for key in self.model_configs.keys():
#             self.model_base[key] = DetectorBase(self.model_configs[key])     
#         try:
#             for key in self.model_map.keys():
#                 res = self.sess.post(self.burl + '/init', data={'model_name': self.model_map[key]}, timeout=30)
#                 if (res.status_code != 200):
#                     raise ValueError
#         except Exception as e:
#             print("Init http api error: ", e)

#     # def __del__(self) -> None:
#     #     self.sess.get(self.base_url + '/destory')

#     def run(self, img_path:str, model_code:str, reg_pts=None):
#         '''
#         params: 
#             img_path: absolute img path
#             reg_pts: points of region (number of points must more than 2)
#         return 
#             same format as Detector class return data
#         '''
#         try:
#             if model_code not in self.model_map.keys():
#                 raise ValueError
#             if not os.path.exists(img_path):
#                 raise ValueError
#             response = self.sess.post(self.burl + '/predict', files={'file': open(img_path, 'rb')}, data={'model_name': self.model_map[model_code]}, timeout=30)
#             res = response.json()["predictions"]
#             if reg_pts is not None and len(reg_pts) >= 3:
#                 res = self.area_select(res, model_code, reg_pts)
#             return res
#         except Exception as e:
#             print("Use post http api error: ", e)
#         return []
    
#     def draw(self, img, det_res:list, model_code):
#         return self.model_base[model_code].draw(img, det_res)
    
#     def area_select(self, det_res:list, model_code, reg_pts=None):
#         return self.model_base[model_code].area_select(det_res, reg_pts)
    