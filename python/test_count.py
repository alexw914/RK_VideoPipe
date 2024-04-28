import cv2, os, datetime
from hyperpyyaml import load_hyperpyyaml
from counters import ObjectCounter
from trackers import BYTETracker

if __name__ == "__main__":


    with open("rknn_model_config.yaml", "r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    video_path = "../assets/videos/person.mp4"
    save_video = True
    save_path = "out"

    model_code = "RYJS"
    start_time = datetime.datetime.now()
    cap     = cv2.VideoCapture(video_path)
    tracker = BYTETracker(cap.get(cv2.CAP_PROP_FPS))
    counter = ObjectCounter()
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty.")
        raise ValueError
    # 初始化视频存储相关类
    if save_video:
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        video_folder = os.path.join(save_path, model_code)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_name = model_code + "_" + start_time.strftime("%Y-%m-%d-%H-%M") + ".avi"
        writer = cv2.VideoWriter(os.path.join(video_folder, video_name), fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]),True)  
    # Set Args
    reg_pts = [[40, frame.shape[0]//2], [frame.shape[1] - 40, frame.shape[0]//2]]
    counter.set_args(hparams["Models"]["WRZS"].labels, reg_pts=reg_pts)
    pre_in, pre_out, pre_count = 0, 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing has been successfully completed.")

        results = hparams["Models"]["WRZS"].run(frame, parse_result=False)
        tracks = tracker.update(results, frame) if results[0] is not None else []
        im0 = counter.start_counting(frame, tracks)

        if save_video:
            writer.write(im0 if im0 is not None else frame)