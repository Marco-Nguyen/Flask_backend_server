import RPi.GPIO as GPIO
import time
# import subprocess
import cv2
import requests
import argparse
import playsound
import time
import os
from pynput.keyboard import Key, Controller

# parser = argparse.ArgumentParser()
# parser.add_argument('--url', type=str, default="http://192.168.100.99:2000/predict", help='last two numbers in IP. Eg: 100.44')
# args = parser.parse_args()
# url = "http://192.168." + args.url + ":2000/predict"
GPIO.setmode(GPIO.BCM)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
filename = "/home/pi/Documents/Flask_backend_server/static/1.jpg"
url = "http://192.168.100.44:2000/predict"

def capture_image(frame):
    cv2.imwrite(filename, frame)
    print("image captured")

def send_to_server():
    time.sleep(1)
    if os.path.exists(filename):
        print("prepare to send to server...")
        my_img = {'image': open(filename, 'rb')}
        print("Here!")
        r = requests.post(url, files=my_img)
        print("Image Sent!")
        url_2 = url.replace("/predict", '/mp3')
        r_2 = requests.get(url)
        print("downloaded!")
#         time.sleep(5)
        open('/home/pi/Documents/message-receive.mp3', 'wb').write(r.content)
        print("Playing Sound")
        playsound.playsound('/home/pi/Documents/message-receive.mp3')
    else:
        print("No image")

while True:
    input_state = GPIO.input(18)
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.imshow('frame', frame)
    
#     keyboard = Controller()
#     key = "q"
    if GPIO.input(24) == True or cv2.waitKey(1) == ord('q'):
       break
#      if cv2.waitKey(1) == ord('q'):
#          break
    
    if input_state == False:
        t0 = time.time()
        capture_image(frame)
        send_to_server()
        print(f"Running time: {time.time() - t0}s")
#         receive_and_play()
    
    # if frame is read correctly ret is True
    

    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
