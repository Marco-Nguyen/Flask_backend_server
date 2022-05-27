import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image

## Test with random number
# tf.random.set_seed(5)
#
# image = tf.zeros([320, 320, 3]) + tf.random.uniform(shape=[320, 320, 3], maxval=255, dtype=tf.float32, seed=10)
# image = tf.cast(image, tf.uint8)
# image = tf.expand_dims(image, axis=0)
# print(type(image))

# print(image)
# detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
# detector_output = detector(image)
# class_ids = detector_output["detection_classes"]

# print(detector_output)
# print(class_ids)

def detect_fn(img, model):

    img = img.resize((320, 320))

    # Preprocessing the image
    # img = np.true_divide(img, 255)
    # img = np.expand_dims(img, axis=0)

    img = tf.convert_to_tensor(img)

    img = tf.cast(img, tf.uint8)
    img = tf.expand_dims(img, axis=0)

    output = model(img)
    results = output["detection_classes"]
    return results

LABELS = ['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'dining table',
'toilet',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush'
]

"""
Box order: [ymin, xmin, ymax, xmax]
"""

img = Image.open(r"D:\Python\Pycharm\Flask\backend_server\static\beach.jpg")
img = img.resize((320, 320))
# img.show()
img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img))
img = tf.cast(img, tf.uint8)
img = tf.expand_dims(img, axis=0)
print(type(img))

my_model= tf.saved_model.load(r"D:\covid-diagnosis-flask\ckpt\detection\saved_model")
results = my_model(img)

labels = []
boxes = []
position = []

# Extract label with scores > threshold
for index, element in enumerate(results['detection_scores'][0] > 0.5):
    if element == True:
        labels.append(LABELS[int(results['detection_classes'][0][index]) - 1])
        boxes.append(results["detection_boxes"][0][index].numpy())

# Find center x coordinate
for box in boxes:
    xc = (box[0] + box[2])/2
    # yc = (box[1] + box[3])/2
    position.append(xc)

dict = {}

# Create dictionary contains label with its corresponding center coordinate
for i, label in enumerate(labels):
    dict[label] = position[i]

print(dict)

# Sort according to center coordinate
# dict = {'key': 0.1, 'dog': 0.9, 'cat': 0.5, 'mouse': 0.4}
a = sorted(dict.items(), key=lambda x: x[1])
print(a)

# print(results['detection_scores'][0] > 0.5)
# print(results['detection_classes'][0])
# print(results["detection_boxes"][0])



