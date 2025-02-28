import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 🔹 Replace with your actual Google Drive File ID
FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"  # <-- Update with your actual ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_PATH = "model.h5"

# 🔹 Function to download and load the model
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
        model = tf.keras.models.load_model(OUTPUT_PATH)
        st.write(f"✅ Model loaded! Expected input shape: {model.input_shape}")
        return model
    else:
        st.error("❌ Model file not found after download.")
        return None

# 🔹 Load the trained model
model = load_model()
if model is None:
    st.stop()

# 🔹 Define class labels
CLASS_NAMES = ["Benign", "Malignant"]

# 🔹 Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 color channels
    image = image.resize((48, 48))  # Resize to match model input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize

    # Flatten if model expects 1D input (modify based on model)
    image = image.flatten().reshape((1, -1))  # Ensure correct input shape

    return image

# 🔹 Streamlit UI
st.title("🔬 Breast Cancer Cell Classification")
st.write("Upload a cell image to classify it as **Benign** or **Malignant**.")

# 🔹 File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Debugging: Print shape before prediction
    st.write(f"📌 Processed Image Shape: {processed_image.shape}")

    # Make prediction
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display results
    st.subheader(f"📌 Prediction: **{CLASS_NAMES[class_index]}**")
    st.write(f"🟢 Confidence: {confidence:.2f}%")
