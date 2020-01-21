import io
import requests
from PIL import Image


def test_app():
    url = "http://localhost:8301/ocr"
    # data = {"image_path": "./text_detect/images/example.jpg"}
    data = {
        "image_path": "/home/bourne/Workstation/AntiGPS/results/google_img/8323675147878602725.jpg"
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=data, headers=headers)
    print(r.json())

    url = "http://localhost:8301/detect"
    r = requests.post(url, json=data, headers=headers)
    print(r.json())

    url = "http://localhost:8301/recognize"
    r = requests.get(url)
    print(r.json())


def test_detect_only():
    url = "http://localhost:8301/detect"
    data = {
        "image_path": "/home/bourne/Workstation/AntiGPS/results/google_img/8323675147878602725.jpg"
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, json=data, headers=headers)
    print(r.json())


if __name__ == "__main__":
    test_app()
    # test_detect_only()
