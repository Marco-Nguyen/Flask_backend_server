from flask import Flask, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2, os, time
import base64
from werkzeug.utils import secure_filename
from flask import jsonify, flash, redirect, send_from_directory
import requests
from PIL import Image

# Disable Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Models
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D
from keras.preprocessing import image


# Identity Block
def identity_block(input, filters, stride=(1, 1)):
    F1, F2, F3 = filters
    x_skip = input

    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=stride, padding='valid')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F2, kernel_size=(3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=stride, padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_skip])

    output = Activation("relu")(x)

    return output


# Convolutional Block
def conv_block(input, filters, stride=(2, 2)):
    F1, F2, F3 = filters
    x_skip = input

    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=stride, padding='valid')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    x_skip = Conv2D(filters=F3, kernel_size=(1, 1), strides=stride, padding='valid')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])

    output = Activation('relu')(x)

    return output


# Resnet50
def ResNet50(input_shape=(224, 224, 3)):
    x_input = Input(input_shape)

    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, filters=[64, 64, 256], stride=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], stride=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], stride=(1, 1))

    x = conv_block(x, filters=[128, 128, 512], stride=(2, 2))
    x = identity_block(x, filters=[128, 128, 512], stride=(1, 1))
    x = identity_block(x, filters=[128, 128, 512], stride=(1, 1))
    x = identity_block(x, filters=[128, 128, 512], stride=(1, 1))

    x = conv_block(x, filters=[256, 256, 1024], stride=(2, 2))
    x = identity_block(x, filters=[256, 256, 1024], stride=(1, 1))
    x = identity_block(x, filters=[256, 256, 1024], stride=(1, 1))
    x = identity_block(x, filters=[256, 256, 1024], stride=(1, 1))
    x = identity_block(x, filters=[256, 256, 1024], stride=(1, 1))
    x = identity_block(x, filters=[256, 256, 1024], stride=(1, 1))

    x = conv_block(x, filters=[512, 512, 2048], stride=(2, 2))
    x = identity_block(x, filters=[512, 512, 2048], stride=(1, 1))
    x = identity_block(x, filters=[512, 512, 2048], stride=(1, 1))

    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    # x = Flatten()(x)
    # x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x)  # multi-class

    model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model


resnet = ResNet50()


# Classifier
def classifier(inputs):
    # x = GlobalMaxPooling2D()(inputs)
    x = Flatten()(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid", name="classification")(x)
    return x


classification_output = classifier(resnet.output)

# cat_dog_model = Model(inputs=resnet.input, outputs=classification_output)

# Load Face Model
# cat_dog_model.load_weights(r"D:\Python\Deep_learning\AS-Study\Task-2-MobileNet-Android\cat-dog.h5")
cat_dog_model = load_model(r"D:\Python\Deep_learning\AS-Study\Task-2-MobileNet-Android\cat-dog.h5")

# Image folder
UPLOAD_FOLDER = r'D:\Python\Pycharm\Flask\backend_server\static\\'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Khởi tạo Flask server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

""" Predict Cat Dog Model"""
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    img = image.img_to_array(img)/255
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # img = preprocess_input(x, mode='caffe')

    preds = model.predict(img)
    if preds <= 0.5:
        return "Predicted labels: " + str(preds[0][0]) + " means Cat"
    else:
        return "Predicted labels: " + str(preds[0][0]) + " means Dog"

def model_predict_2(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    # img /= 255
    img = np.true_divide(img, 255)
    img = np.expand_dims(img, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # img = preprocess_input(x, mode='caffe')

    preds = model.predict(img)
    if preds <= 0.5:
        return str(preds[0][0]), "Cat"
    else:
        return str(preds[0][0]), "Dog"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/uploads/<path:filename>')
# def download_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename, as_attachment=True)

@app.route('/', methods=['POST', 'GET'])
@cross_origin(origin='*')
def catvdog():
    # if request.method == 'POST':
        # Read image from client
        # f = request.files["file"]
        #
        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        # print(request.method)

        # if request.method == 'POST':
        #     # check if the post request has the file part
        #     if 'file' not in request.files:
        #         flash('No file part')
        #         return redirect(request.url)
        #     file = request.files['file']
        #     # if user does not select file, browser also
        #     # submit a empty part without filename
        #     if file.filename == '':
        #         flash('No selected file')
        #         return redirect(request.url)
        #     if file and allowed_file(file.filename):
        #         filename = secure_filename(file.filename)
        #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #         print(str(file))

        # Predict if it is a cat or dog
    # print("Posted file: {}".format(request.files['file']))
    # file = request.files['file']
    # files = {'file': file.read()}
    # r = requests.post("http://127.0.0.0:5000/upload/", files = files)
    # if r.ok:
    #     print("File uploaded!")
    # else:
    #     print("Error uploading file!")
    t0 = time.time()
    img_path = r"D:\Python\Pycharm\Flask\capture_img\images\img.jpg"
    if os.path.exists(img_path):
        preds = model_predict(img_path, cat_dog_model)
        results = preds + " in " + str(time.time() - t0) + "s"
        os.remove(r"D:\Python\Pycharm\Flask\capture_img\images\img.jpg")
        return results
    else:
        return "No image is captured"

@app.route("/catvdog", methods=["POST"])
def process_image():
    t0 = time.time()
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    if img is not None:
        pred, label = model_predict_2(img, cat_dog_model)
        # os.remove(r"D:\Python\Pycharm\Flask\capture_img\images\img.jpg")
        return jsonify({'msg': 'success', 'response time': time.time() - t0, 'prediction': pred, 'label': label})


if __name__ == "__main__":
    app.run(debug=True)


# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
