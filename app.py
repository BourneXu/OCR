# -*- coding:utf-8 -*-

import os
import io
from datetime import datetime
from ocr import OCR
from PIL import Image
from types import SimpleNamespace
from flask import Flask, request, jsonify

app = Flask(__name__)
app.secret_key = os.urandom(12)


@app.route("/ocr", methods=["POST"])
def ocr():
    image_path = request.json["image_path"]
    r = engine.run(image_path)
    return jsonify(r)


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
    engine = OCR(opt)
    app.run(host="localhost", port=8301, debug=True)

