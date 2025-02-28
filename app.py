import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Replace with your actual Google Drive File ID
FILE_ID = "https://drive.google.com/file/d/124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X/view?usp=sharing"
URL = f"https://drive.google.com/uc?export=download&id={https://drive.google.com/file/d/124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X/view?usp=sharing}"
OUTPUT_PATH = "model.h5"

# Function to download the model from Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists(OUTPUT_PATH):  # Check if model already exists
        st.write("📥 Downloading model...")
        try:
            gdown.download(URL, OUTPUT_PATH, quiet=False)
            st.write("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None

    if os.path.exists(OUTPUT_PATH):
        return tf.keras.models.load_model(OUTPUT_PATH)
    else:
        st.error("❌ Model file not found after download.")
        return None

# Load model
model = load_model()
if model is None:
    st.stop()

# Define class labels
CLASS_NAMES = ["Benign", "Malignant"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Breast Cancer Cell Classification")
st.write("Upload a cell image to classify it as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: **{CLASS_NAMES[class_index]}**")
    st.write(f"Confidence: {confidence:.2f}%")
