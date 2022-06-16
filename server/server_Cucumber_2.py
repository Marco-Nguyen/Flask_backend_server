import cv2
from flask import Flask, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS, cross_origin
import os, time
from PIL import Image
import numpy as np
from gtts import gTTS
from werkzeug.utils import secure_filename


import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

# Disable Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SETUP
IMG_SIZE = 416
LABELS = ['binhthuong', 'phantrang', 'suongmai']

"""
Box order: [ymin, xmin, ymax, xmax]
"""

# Khởi tạo Flask server Backend
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_path = r"D:\Python\Pycharm\AI-in-agriculture\weights\Smart-agriculture\best_saved_model"
# UPLOAD_FOLDER = "static\dog-3.jpg"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
def load_tf_model(path_to_model):
    # model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    start = time.time()
    model = tf.saved_model.load(path_to_model)
    end = time.time()
    print("Timeout: ", end - start)
    # model
    return model

# Predict
# @tf.function
def detect_fn(img, model):

    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Preprocessing the image
    # img = np.true_divide(img, 255)
    # img = np.expand_dims(img, axis=0)

    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img))

    # img = tf.cast(img, tf.uint8)
    img = tf.expand_dims(img, axis=0)

    output = model(img)
    # results = output["detection_classes"]
    return output

def results_handling(results):
    labels = []
    # boxes = []
    position = []
    dict = {}
    for index, element in enumerate(results['detection_scores'][0] > 0.5):
        if element == True:
            # print(element)
            labels.append(LABELS[int(results['detection_classes'][0][index]) - 1])
            boxes = results["detection_boxes"][0][index].numpy()
            position.append((boxes[0] + boxes[2])/2)

    # for box in boxes:
    #     xc = (box[0] + box[2]) / 2
    #     position.append(xc)

    for i, label in enumerate(labels):
        dict[label] = position[i]

    sorted_dict = sorted(dict.items(), key=lambda x: x[1])
    message_EN = "From left to right: "
    # message_VN = "Từ trái sang phải: "
    for element in sorted_dict:
        message_EN += f"{element[0]} "
        # message_VN += f"{element[0]} "

    return message_EN

def generate():
    path = "message.mp3"
    with open(path, 'rb') as fmp3:
        data = fmp3.read(1024)
        while data:
            yield data
            data = fmp3.read(1024)

model = load_tf_model(model_path)

# Download mp3 files
@app.route('/mp3', methods=['GET'])
def get_mp3():
    return Response(generate(), mimetype="audio/mpeg3")

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
        results = detect_fn(img, model)
        print(results)
        # message = results_handling(results)
        # gTTS(text=message, lang="en").save("message.mp3")

        return jsonify({'status': 'success', 'response time': time.time() - t0, "message": results})

        # return Response(generate(), mimetype="audio/mpeg3")
    else:
        return "EMPTY"

# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
