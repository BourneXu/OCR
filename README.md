# Scene-OCR
CRAFT: text detection + TPS-ResNet-BiLSTM-Attn: text recognition

This repo combines two methods which are most popular and having best performace to meet my demand in Scene-OCR. I have modified two forked repos and rewrite a Flask service for local users to use detection, recognition and OCR conveniently.

## How to use
1. Install all dependency. If there are still some missing packages, you can check original forked repos.
```bash
pip install -r requirements.txt
```
2. It is better to run app under GPU.
```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```
3. Try it by yourself. You can send `POST` requests to `http://localhost:8301/ocr` with your image absolute path. See more details in `test_app.py`.

## Text Detection
This part is forked by [clovaai/CRAFT-pyotrch](https://github.com/clovaai/CRAFT-pytorch).

Loaded model is `models/craft_mlt_25k.pth`. You can download it from [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ).

## Text Recognition
This part is forked by [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

Loaded model is `models/TPS-ResNet-BiLSTM-Attn.pth`. You can download it from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)


## Notice
The link for both models are supported by original repos. If there are any issues, you could post issues in there.