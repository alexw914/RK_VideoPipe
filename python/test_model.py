import cv2, os
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

if __name__ == "__main__":
    
    with open("rknn_model_config.yaml", "r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)
          
    model_codes = list(hparams["Models"].keys())
    img_folder = "../assets/images"
    save_folder = "out"
    img_files = os.listdir(img_folder)
    Path(save_folder).mkdir(exist_ok=True)
    for file in img_files:
        img = cv2.imread(os.path.join(img_folder, file))
        for code in model_codes:
            save_path = os.path.join(save_folder, code)
            Path(save_path).mkdir(exist_ok=True)
            res = hparams["Models"][code].run(img)
            out_img = hparams["Models"][code].draw(img, res)
            cv2.imwrite(os.path.join(save_path, file), out_img)
          
       



