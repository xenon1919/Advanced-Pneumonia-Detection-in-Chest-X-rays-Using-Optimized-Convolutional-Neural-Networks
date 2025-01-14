

# Advanced Pneumonia Detection in Chest X-rays Using Optimized Convolutional Neural Networks

## Description
This project is a web-based application designed to detect pneumonia in chest X-ray images using deep learning. It utilizes a pre-trained Convolutional Neural Network (CNN) model, processes the uploaded images, and provides a prediction with confidence levels.

## Features
- Upload a chest X-ray image to detect pneumonia.
- Real-time image preprocessing and normalization.
- Display prediction results with confidence levels.
- User-friendly interface using Streamlit.

## Technologies Used
- **Python**: Backend development.
- **Streamlit**: Web app creation.
- **TensorFlow/Keras**: Deep learning model for pneumonia detection.
- **NumPy**: Numerical operations and image manipulation.

## How to Run the App Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/xenon1919/Advanced-Pneumonia-Detection-in-Chest-X-rays-Using-Optimized-Convolutional-Neural-Networks.git
   cd pneumonia-detection
   ```

2. Set up the virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Windows: `env\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501` to access the application.

## License
This project is licensed under the MIT License.

---
