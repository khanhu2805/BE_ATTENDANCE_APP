import os
from flask import Flask, request
import base64
from io import BytesIO
from PIL import Image, ImageFile
from ultralytics import YOLO
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import datetime
import torch

app = Flask(__name__)

segmentor = SelfiSegmentation()
bg_image = cv2.imread('background.jpg')


def check_gpu_memory():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        mem_reserved = torch.cuda.memory_reserved(device)
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_free = mem_reserved - mem_allocated
        return mem_free > 0
    else:
        return False


@app.get('/')
def homepage():
    return 'Python Backend API'


@app.post('/')
def detect():
    student_code = request.get_json()['student_code']
    image = request.get_json()['image']
    model = YOLO(
        f"{student_code[:2]}/{student_code[4:6]}/{int(student_code[6:]) // 50 * 50}-{int(student_code[6:]) // 50 * 50 + 49}.pt")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = image.encode("ascii")
    rem = len(image) % 4
    if rem > 0:
        image += b"=" * (4 - rem)
    img_bytes = base64.b64decode(image)
    # Mở ảnh từ bytes
    img = Image.open(BytesIO(img_bytes))
    image = img.convert('RGB')
    date = datetime.datetime.now()
    date_str = str(date).replace(":", "_")
    image.save(f"test{date_str}.png")
    video_capture = cv2.imread(f"test{date_str}.png")
    h, w = video_capture.shape[:2]
    bg_image_resized = cv2.resize(bg_image, (w, h))
    remove_background = segmentor.removeBG(video_capture, bg_image_resized, cutThreshold=0.8)
    cv2.imwrite(f"test{date_str}.png", remove_background)
    if check_gpu_memory():
        result = model.predict(source=f"test{date_str}.png")[0]
    else:
        # Chuyển đổi sang CPU và thực hiện dự đoán trên CPU
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        device = torch.device('cpu')
        model.to(device)
        result = model.predict(source=f"test{date_str}.png")[0]
    os.remove(f"test{date_str}.png")
    if result.probs.top1conf > 0.8:
        return result.names[result.probs.top1]
    else:
        return 'Khong nhan dien duoc'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
