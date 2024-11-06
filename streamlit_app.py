import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = CNN()  # Use the custom CNN class you defined earlier
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("path_to_your_trained_model.h5")  # Make sure to replace this with your actual model path
    return model

model = load_model()

# Define image preprocessing function
def preprocess_image(uploaded_img, img_size=(224, 224)):
    img = Image.open(uploaded_img)
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Define the prediction function
def predict_image(img_array):
    prediction = model.predict(img_array)
    if prediction[0] < 0.5:
        return "No Human Detected"
    else:
        return "Human Detected"

# Streamlit App Layout
st.title("Thermal Image Human Detection")
st.write("Upload an image to classify if it contains a human or not.")

# File uploader for images
uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    img_array = preprocess_image(uploaded_file)
    prediction = predict_image(img_array)
    
    # Display prediction
    st.write("Prediction:", prediction)
