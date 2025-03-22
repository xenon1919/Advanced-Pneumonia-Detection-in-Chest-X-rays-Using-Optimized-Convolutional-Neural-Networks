import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=1cMnej9Z3SCXQrbqgQGrkRDSoWsBT5k-P"

# 🔹 Model file path
MODEL_PATH = "model.h5"

# 🔄 Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Downloading model... Please wait."):
        gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)

# 🧠 Load the trained model
@st.cache_resource
def load_trained_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None  # Prevents crash

model = load_trained_model()

# 🎨 Streamlit UI Enhancements
st.markdown("<h1 style='text-align: center;'>🩺 Pneumonia Detection</h1>", unsafe_allow_html=True)
st.write("## Upload a chest X-ray image to check for pneumonia. 🔬")

# 📤 File uploader
uploaded_file = st.file_uploader("📂 Choose an X-ray image", type=["jpg", "jpeg", "png"])

# 🖼️ Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize
    return image_array

# 🏥 Model Prediction Section
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if model is not None:
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        confidence = np.max(prediction)
        class_label = np.argmax(prediction)

        labels = {0: "🟢 Normal", 1: "🔴 Pneumonia"}

        # 🏷️ Confidence threshold for accurate results
        result = "❓ Uncertain" if confidence < 0.6 else labels[class_label]

        # 📊 Display results with styling
        st.markdown(f"<h2 style='text-align: center;'>🏥 Prediction: <b>{result}</b></h2>", unsafe_allow_html=True)
        st.write(f"**🔬 Confidence Level:** {confidence:.2f}")

        # ✅ Fixed: Convert float to percentage for progress bar
        st.progress(int(confidence * 100))

    else:
        st.error("🚨 Model failed to load. Please check your model file.")
