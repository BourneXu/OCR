import os
import time
from types import SimpleNamespace

from text_detect.detection import Detection
from text_recognize.recognition import Recognition


class OCR:
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(opt.detect_result_folder, exist_ok=True)
        self.detection = Detection(
            trained_model=opt.detect_trained_model, result_folder=opt.detect_result_folder
        )
        self.recognition = Recognition(
            image_folder=opt.recognize_image_folder,
            saved_model=opt.recognize_saved_model,
            Transformation=opt.recognize_Transformation,
            FeatureExtraction=opt.recognize_FeatureExtraction,
            SequenceModeling=opt.recognize_SequenceModeling,
            Prediction=opt.recognize_Prediction,
        )

    def __check_folder(self):
        if len(os.listdir(self.opt.detect_result_folder)):
            os.system(f'rm {self.opt.detect_result_folder.rstrip("/")}/*')

    def run(self, image_path):
        self.__check_folder()
        self.num, self.locations, self.image_size = self.detection.TextDetect(image_path)
        result = self.recognition.TextRecognize()
        for k, v in result.items():
            v["location"] = self.locations[k]
            result[k] = v
        return {
            "height": self.image_size[0],
            "width": self.image_size[1],
            "text": list(result.values()),
        }

    def detect_only(self, image_path):
        self.__check_folder()
        self.num, self.locations, self.image_size = self.detection.TextDetect(image_path)
        return {"detect_result_folder": self.opt.detect_result_folder}

    def recognize_only(self):
        result = self.recognition.TextRecognize()
        for k, v in result.items():
            v["location"] = self.locations[k]
            result[k] = v
        return {
            "height": self.image_size[0],
            "width": self.image_size[1],
            "text": list(result.values()),
        }


if __name__ == "__main__":
    path_abs = os.path.dirname(os.path.abspath(__file__))
    opt = SimpleNamespace()
    opt.detect_trained_model = f"{path_abs}/models/craft_mlt_25k.pth"
    opt.detect_result_folder = f"{path_abs}/images/box/"
    opt.recognize_image_folder = f"{path_abs}/images/box/"
    opt.recognize_saved_model = f"{path_abs}/models/TPS-ResNet-BiLSTM-Attn.pth"
    opt.recognize_Transformation = "TPS"
    opt.recognize_FeatureExtraction = "ResNet"
    opt.recognize_SequenceModeling = "BiLSTM"
    opt.recognize_Prediction = "Attn"
    opt.test_image = f"{path_abs}/text_detect/images/example.jpg"

    test = OCR(opt)
    # r = test.run(opt.test_image)
    r = test.run("/home/bourne/Workstation/AntiGPS/results/google_img/8323675147878602725_2.jpg")
    print(r)
