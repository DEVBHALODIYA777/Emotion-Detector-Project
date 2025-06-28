# Emotion Detection Web App – Real Website Look (Streamlit)

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

# Emotion labels and emojis
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😱',
    'Happy': '😄',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}

# Page Config and Custom CSS
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f3f4f6;
        }
        .hero {
            text-align: center;
            padding: 1.5em 0 0.5em 0;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border-radius: 10px;
            margin-bottom: 2em;
        }
        .hero h1 {
            font-size: 3em;
            margin: 0.2em 0;
        }
        .section {
            background-color: white;
            padding: 2em;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2em;
        }
        .emotion-result {
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-top: 1em;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 1em;
            padding: 0.6em 1.2em;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>😃 Real-Time Emotion Detection</h1>
    <p>Upload an image or use your webcam to predict human emotions using AI</p>
</div>
""", unsafe_allow_html=True)

# Helper function
def predict_emotion_from_array(img_array):
    img_resized = cv2.resize(img_array, (48, 48))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    prediction = model.predict(img_input)[0]
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]
    confidence = prediction[emotion_idx] * 100
    return emotion, confidence

# Upload Image Section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📁 Upload an Image")
uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    image = Image.open(uploaded_file).convert('L')  # grayscale
    image_np = np.array(image)
    emotion, confidence = predict_emotion_from_array(image_np)
    st.markdown(f"<div class='emotion-result'>Detected Emotion: {emotion_emojis[emotion]} {emotion}<br>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Webcam Section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📷 Live Webcam Detection")
run_webcam = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run_webcam:
    camera = cv2.VideoCapture(0)
    while run_webcam:
        success, frame = camera.read()
        if not success:
            st.error("Webcam not accessible.")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray_frame, (48, 48))
        emotion, confidence = predict_emotion_from_array(face_resized)
        label = f"{emotion_emojis[emotion]} {emotion} ({confidence:.1f}%)"
        frame = cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    camera.release()
else:
    st.info("Check the box above to activate webcam.")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style='margin-top:2em;'>
<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit & TensorFlow</p>
""", unsafe_allow_html=True)
