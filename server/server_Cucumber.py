import cv2
from flask import Flask, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS, cross_origin
import os, time, serial
from PIL import Image
import numpy as np
from gtts import gTTS
from werkzeug.utils import secure_filename
import io

import torch
import torchvision.transforms as transforms

from Flask.backend_server.Flask_backend_server.yolo_utils.inference import ONNX_engine

# Disable Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SETUP
IMG_SIZE = 416


"""
Box order: [ymin, xmin, ymax, xmax]
"""

# Khởi tạo Flask server Backend
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_path = r"D:\Python\Pycharm\AI-in-agriculture\weights\Smart-agriculture\updated-1.pt"
UPLOAD_FOLDER = "static\dog-3.jpg"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ser = serial.Serial(port='COM10', baudrate=9600)

# Load models
def load_model_pt(model_path):
    start = time.time()
    model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path)

    ## Set up params
    model.conf = 0.7  # NMS confidence threshold
    model.iou = 0.5  # NMS IoU threshold
    model.classes = [0, 1, 2]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = True  # NMS multiple labels per box
    model.max_det = 10  # maximum number of detections per image
    end = time.time()
    print("Timeout: ", end - start)
    return model

def load_model_onnx(model_path):
    model = ONNX_engine()
    return model

model = load_model_pt(model_path)

# Predict
# @tf.function
def detect_fn_pt_audio(model, image):
    # print(image)
    # to_tensor = transforms.ToTensor()
    # img = to_tensor(image)
    # img = img.numpy()
    # print(type(img))
    diagnosis = []
    results = model(image, size=416)
    if results.xyxy[0].nelement() != 0:
        if 1 in results.xyxy[0][:, 5]:
            diagnosis.append(1)
        if 2 in results.xyxy[0][:, 5]:
            diagnosis.append(2)
        if len(diagnosis) == 0:
            diagnosis.append(0)
    else:
        diagnosis.append(-1)

    message_EN = "WARNING: "
    if len(diagnosis) == 2:
        message_EN += "your cucumbers have both diseases on the leaves"
    elif results == [1]:
        message_EN += "your cucumbers have powdery mildew on the leaves"
    elif results == [2]:
        message_EN += "your cucumbers have downy mildew on the leaves"
    else:
        message_EN = "your cucumbers are healthy"

    message_VN = "Cảnh báo: "
    if len(diagnosis) == 2:
        message_VN += "cây dưa chuột bị cả hai bệnh!"
    elif diagnosis == [1]:
        message_VN += "cây dưa chuột bị bệnh phấn trắng!"
    elif diagnosis == [2]:
        message_VN += "cây dưa chuột bị bệnh sương mai!"
    elif diagnosis == [0]:
        message_VN = "cây dưa chuột khỏe mạnh"
    else:
        message_VN = "Không nhận diện được lá dưa chuột"
    return message_EN, message_VN


# def detect_fn_onnx(model, image):
#     url = "http://192.168.55.106:2000/predict"
#     url_2 = url.replace("/predict", '/mp3')
#     r_2 = requests.get(url)
#     return model.run(weights=weights_path, source=image, view_img=False)

def detect_fn_pt_dict(model, image):
    diagnosis = ""
    results = model(image, size=416)
    if results.xyxy[0].nelement() != 0:
        if 1 in results.xyxy[0][:, 5]:
            diagnosis += "1"
        if 2 in results.xyxy[0][:, 5]:
            diagnosis += "2"
        if len(diagnosis) == 0:
            diagnosis += "0"
    else:
        diagnosis += "-1"
    return  diagnosis


# Create mp3
def generate():
    path = "message.mp3"
    with open(path, 'rb') as fmp3:
        data = fmp3.read(1024)
        while data:
            yield data
            data = fmp3.read(1024)

def create_txt(results):
    path = "../messages/results.txt"
    with open(path, 'w') as text_file:
        text_file.write(results)
        data = text_file.readline()
        yield data

# Download mp3 files
@app.route('/mp3', methods=['GET'])
def get_mp3():
    return Response(generate(), mimetype="audio/mpeg3")

# Json response
# @app.route('/json', methods=['GET'])
# def get_json():
#     return Response(response=create_txt(), mimetype="application/txt")

@app.route('/predict', methods=['POST'])
def prediction():
    t0 = time.time()
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    # files = {'media': open(path_img, 'rb')}
    # requests.post(url, files=files)
    # global model

    if img is not None:
        results = detect_fn_pt_dict(model, img)
        # return results
        return Response(create_txt(results), mimetype="application/txt")
        # message = results_handling(results)

        # EN, VN = detect_fn_pt(model, img)
        # gTTS(text=VN, lang="vi").save("message.mp3")

        # if message != '0':
        #     ser = serial.Serial(port='COM10', baudrate=9600)
        #     time.sleep(2)
        #     print(ser)
        #     ser.write(message.encode())
        #     return jsonify({'status': 'success', 'response time': time.time() - t0, "message": message})
        # else:
        #     return jsonify({'status': 'success', 'response time': time.time() - t0, "message": message})

        # return Response(generate(), mimetype="audio/mpeg3")
    else:
        return "EMPTY"

# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
