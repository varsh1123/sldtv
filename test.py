import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pyttsx3
from playsound import playsound
import tensorflow as tf

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
engine.setProperty('voice', 'ta')  # Set Tamil voice

# Define the labels and corresponding audio paths
labels_and_audio = {
    0: {"label": "Hello", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\hellota.mp3"},
    1: {"label": "Love you", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\i love you ta.mp3"},
    2: {"label": "No", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\No ta.mp3"},
    3: {"label": "Okay", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\okay ta.mp3"},
    4: {"label": "Please", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\please ta.mp3"},
    5: {"label": "Thank you", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\thank you ta.mp3"},
    6: {"label": "Yes", "audio_path": r"C:\Users\USER\OneDrive\Desktop\sign language detection\yesta.mp3"}
}

# Load the Keras CNN model
cnn_model = tf.keras.models.load_model("F:\Downloads\keras_model.h5")

# Define the LSTM model
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(None, cnn_model.output_shape[1])),
    tf.keras.layers.Dense(len(labels_and_audio), activation='softmax')
])
# Add any additional layers to the LSTM model as needed

# Initialize the HandDetector
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Main loop
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
        
        # Preprocess the image for the CNN model
        imgPreprocessed = cv2.resize(imgWhite, (224, 224))
        imgPreprocessed = np.expand_dims(imgPreprocessed, axis=0)
        imgPreprocessed = imgPreprocessed / 255.0

        # Use the CNN model for feature extraction
        cnn_features = cnn_model.predict(imgPreprocessed)

        # Reshape the features for LSTM input
        lstm_input = np.reshape(cnn_features, (1, cnn_features.shape[0], cnn_features.shape[1]))

        # Use LSTM model for sequence learning
        prediction_probs = lstm_model.predict(lstm_input)[0]
        prediction = np.argmax(prediction_probs)
        label = labels_and_audio[prediction]["label"]
        audio_path = labels_and_audio[prediction]["audio_path"]

        # Speak the detected sign label
        engine.say(label)
        engine.runAndWait()

        # Play audio based on index
        playsound(audio_path)

        # Display the label on the image
        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
