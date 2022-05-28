import cv2
from flask import Flask, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS, cross_origin
import os, time
from PIL import Image
import numpy as np
from gtts import gTTS
from keras import Input, Model
from keras.layers import Activation, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from werkzeug.utils import secure_filename


import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

# SETUP
IMG_SIZE = 320
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

"""
Box order: [ymin, xmin, ymax, xmax]
"""

# Khởi tạo Flask server Backend
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_path = r"D:\covid-diagnosis-flask\ckpt\brain-tumor\brain-tumor-multiple-categories\saved_model"
UPLOAD_FOLDER = r"D:\Python\Pycharm\AS-Study\Task-7-Brain-Tumor\brain-tumor-multiple-categories\Testing\glioma_tumor\\"

# Define model
def create_model(input_shape=(224, 224, 3)):
    x_input = Input(input_shape)

    x = Conv2D(32, kernel_size=(5, 5), strides=(2, 2))(x_input)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.silu)(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.silu)(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(128, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.silu)(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(256, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.silu)(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)

    # x = Conv2D(256, kernel_size=(3, 3))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2))(x)

    # Classifier
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x_output = Dense(4, activation="softmax", name="classification")(x)

    model = Model(inputs=x_input, outputs=x_output)
    return model
#
# model = create_model((224, 224, 3))
# model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

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
    image = np.array(img)
    # image = cv2.imread(UPLOAD_FOLDER + str(img))
    image = cv2.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)

    prediction = model(image)
    pred_label = labels[prediction.argmax()]

    return pred_label

# def results_handling(results):
#     labels = []
#     # boxes = []
#     position = []
#     dict = {}
#     for index, element in enumerate(results['detection_scores'][0] > 0.5):
#         if element == True:
#             labels.append(LABELS[int(results['detection_classes'][0][index]) - 1])
#             boxes = results["detection_boxes"][0][index].numpy()
#             position.append((boxes[0] + boxes[2])/2)
#
#     # for box in boxes:
#     #     xc = (box[0] + box[2]) / 2
#     #     position.append(xc)
#
#     for i, label in enumerate(labels):
#         dict[label] = position[i]
#
#     sorted_dict = sorted(dict.items(), key=lambda x: x[1])
#     message_EN = "From left to right: "
#     # message_VN = "Từ trái sang phải: "
#     for element in sorted_dict:
#         message_EN += f"{element[0]} "
#         # message_VN += f"{element[0]} "
#
#     return message_EN

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
        # message = results_handling(results)
        # gTTS(text=message, lang="en").save("message.mp3")

        return jsonify({'status': 'success', 'response time': time.time() - t0, "message": results})

        # return Response(generate(), mimetype="audio/mpeg3")
    else:
        return "EMPTY"

# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
