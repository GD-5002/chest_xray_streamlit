import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model("chest_xray_model.h5")

# Set page config
st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")

# Title
st.title("🩺 Chest X-Ray Pneumonia Detection")
st.markdown("Upload a chest X-ray and let the model detect whether it's **Pneumonia** or **Normal**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show the image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 224, 224, 1)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show results
    st.markdown(f"### 🧠 Prediction: **{result}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
