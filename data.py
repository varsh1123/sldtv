import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import pyttsx3  # Add for text-to-speech

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed as desired

# Initialize HandDetector
detector = HandDetector(maxHands=1)

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\USER\OneDrive\Desktop\sign language detection\Data\Okay"

# Get language selection
language_code = input("Select language (en for English): ")  # You can add more language options

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

        # Speak the captured sign
        engine.say("Captured sign in " + language_code + " folder")  # Adjust the spoken text as needed
        engine.runAndWait()

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cv2.destroyAllWindows()
