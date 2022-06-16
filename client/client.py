from __future__ import print_function
import requests
import argparse
import playsound

#url = "100.99"
# IMG_PATH = r"static//"
IMG_PATH = r"D:\Python\Pycharm\Flask\backend_server\Flask_backend_server\static\\"
path_img = "person.jpg"

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, default="http://192.168.100.99:2000/predict", help='last two numbers in IP. Eg: 100.44')
parser.add_argument('--img', type=str, default=path_img, help='image name')
args = parser.parse_args()

# Image POST request
url = "http://192.168." + args.url + ":2000/predict"
img = args.img

my_img = {'image': open(IMG_PATH + img, 'rb')}
r = requests.post(url, files=my_img)
print("Image Sent!")

# convert server response into JSON format.
# print(r.json())

## Download file
url_2 = url.replace("/predict", '/json')
r_2 = requests.get(url)
# r_3 = requests.get(url_2)
print(r_2)
# print(r.headers.get('content-type'))

## Play sound in RPI
# open('message-receive.mp3', 'wb').write(r.content)
# print("Playing Sound")
# playsound.playsound('message-receive.mp3')

# Read txt file
with open(r"D:\Python\Pycharm\Flask\backend_server\Flask_backend_server\messages\results.txt", 'r') as text_file:
    result = text_file.readline()
    print(result)