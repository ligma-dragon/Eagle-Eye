import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model

# Paths for dataset and model
DATASET_PATH_test = "/Users/adithyamallya/Documents/PESU Academy/CSE/Sem3/PESU I:O/final_assignment/Thermal_Overhead.v8i.yolov8/test"
MODEL_PATH = "/Users/adithyamallya/Documents/PESU Academy/CSE/Sem3/PESU I:O/final_assignment/model.keras"

# Load the trained model
model = load_model(MODEL_PATH)

# Load and plot training history
def plot_accuracy_and_loss(history):
    epochs = range(1, len(history['accuracy']) + 1)

    # Plot accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(epochs, history['accuracy'], 'bo-', label='Training accuracy')
    ax1.plot(epochs, history['val_accuracy'], 'r^-', label='Validation accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(epochs, history['loss'], 'bo-', label='Training loss')
    ax2.plot(epochs, history['val_loss'], 'r^-', label='Validation loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    st.pyplot(fig)

# Display the accuracy and loss curves
st.title("CNN Model Training Results")
st.subheader("Training and Validation Curves")

# Replace this with your history if you have saved it
history = {
    'accuracy': [0.85, 0.92],
    'val_accuracy': [0.88, 0.91],
    'loss': [0.5, 0.4],
    'val_loss': [0.45, 0.35]
}
plot_accuracy_and_loss(history)

# Function to preprocess and predict on an uploaded image
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    class_label = "Human Detected" if prediction > 0.5 else "No Human Detected"
    
    # Display image and prediction
    st.image(img, caption="Selected Image", use_column_width=True)
    st.write(f"Prediction: {class_label} ({prediction:.2f})")

# Image testing section
st.subheader("Test the Model on an Image")
uploaded_file = st.file_uploader("Choose an image from the dataset", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    predict_image(uploaded_file)
