import test_button
import RPi.GPIO as GPIO
from subprocess import call
import argparse

GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# parser = argparse.ArgumentParser()
# parser.add_argument('--url', type=str, default="http://192.168.100.99:2000/predict", help='last two numbers in IP. Eg: 100.44')
# args = parser.parse_args()
# url = "http://192.168." + args.url + ":2000/predict"

GPIO.setmode(GPIO.BCM)

while True:
    input_state = GPIO.input(24)
    print(input_state)
    if input_state == False:
#         call(["python", '/home/pi/Documents/test_button.py', f'{args.url}'])
        call(["python", '/home/pi/Documents/test_button.py'])
#         print("hello")
#     else:
#         print("Idle")
#         input_state = GPIO.input(24)
