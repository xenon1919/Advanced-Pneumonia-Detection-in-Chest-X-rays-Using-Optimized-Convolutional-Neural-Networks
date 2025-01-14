import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

# Streamlit app
st.title("Pneumonia Detection App")
st.write("Upload a chest X-ray image to check for pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

# Preprocess function
def preprocess_image(image, target_size=(224, 224)):
    # Convert grayscale to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize and preprocess
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize
    return image_array

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(preprocessed_image)
    confidence = np.max(prediction)
    class_label = np.argmax(prediction)

    # Class mapping
    labels = {0: "Normal", 1: "Pneumonia"}
    
    # Post-process the prediction
    if confidence < 0.6:  # Set confidence threshold
        result = "Uncertain"
    else:
        result = labels[class_label]

    # Display the results
    st.write(f"### Prediction: **{result}**")
    if result == "Uncertain":
        st.write(f"Confidence: **{confidence:.2f}** — The model is not confident enough.")
    else:
        st.write(f"Confidence: **{confidence:.2f}**")

    # Add confidence progress bar
    st.progress(int(confidence * 100))
