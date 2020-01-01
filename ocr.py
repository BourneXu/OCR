import os
import time
from types import SimpleNamespace

from text_detect.detection import Detection
from text_recognize.recognition import Recognition


class OCR:
    def __init__(self, opt):
        self.opt = opt
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

    def run(self, image_path):
        num = self.detection.TextDetect(image_path)
        result = self.recognition.TextRecognize()
        return result


if __name__ == "__main__":
    path_abs = os.path.dirname(os.path.abspath(__file__))
    opt = SimpleNamespace()
    opt.detect_trained_model = f"{path_abs}/text_detect/craft_mlt_25k.pth"
    opt.detect_result_folder = f"{path_abs}/images/"
    opt.recognize_image_folder = f"{path_abs}/images/"
    opt.recognize_saved_model = f"{path_abs}/text_recognize/TPS-ResNet-BiLSTM-Attn.pth"
    opt.recognize_Transformation = "TPS"
    opt.recognize_FeatureExtraction = "ResNet"
    opt.recognize_SequenceModeling = "BiLSTM"
    opt.recognize_Prediction = "Attn"
    opt.test_image = f"{path_abs}/text_detect/images/example.jpg"

    test = OCR(opt)
    r = test.run(opt.test_image)
    print(r)
