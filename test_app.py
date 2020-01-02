import io
import requests
from PIL import Image


def test_app():
    url = "http://localhost:8301/ocr"
    # data = {"image_path": "./text_detect/images/example.jpg"}
    data = {
        "image_path": "/home/bourne/Workstation/AntiGPS/results/images/--HTQAvDCOs0w11zhyHWow.jpg"
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=data, headers=headers)
    print(r.json())


if __name__ == "__main__":
    test_app()
