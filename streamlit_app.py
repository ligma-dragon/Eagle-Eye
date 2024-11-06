import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Paths for dataset and model
DATASET_PATH_train = "/Users/adithyamallya/Documents/PESU_Academy/CSE/Sem3/PESU_IO/final_assignment/Thermal_Overhead.v8i.yolov8/train"
DATASET_PATH_valid = "/Users/adithyamallya/Documents/PESU_Academy/CSE/Sem3/PESU_IO/final_assignment/Thermal_Overhead.v8i.yolov8/valid"
MODEL_PATH = "/Users/adithyamallya/Documents/PESU_Academy/CSE/Sem3/PESU_IO/final_assignment/model.keras"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load data for demonstration
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH_train, target_size=(224, 224), batch_size=1, class_mode='binary', shuffle=True)

# Display model summary
st.title("Thermal Overhead Human Detection Model")
st.header("Model Summary")
st.text(model.summary())

# Plot accuracy and loss curves
st.header("Training and Validation Accuracy/Loss")

# Assuming 'history' data is pre-saved or loaded
def plot_accuracy_and_loss(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy
    ax1.plot(epochs, history.history['accuracy'], 'bo-', label='Training accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], 'r^-', label='Validation accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(epochs, history.history['loss'], 'bo-', label='Training loss')
    ax2.plot(epochs, history.history['val_loss'], 'r^-', label='Validation loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    st.pyplot(fig)

# Placeholder for history data (uncomment and replace with your actual history data if available)
# plot_accuracy_and_loss(history)

# Allow the user to test images from the dataset
st.header("Test the Model with Images")

# Select an image from the training dataset
image_files = [os.path.join(DATASET_PATH_train, img) for img in os.listdir(DATASET_PATH_train) if img.endswith(('.png', '.jpg', '.jpeg'))]
selected_image = st.selectbox("Choose an image to test:", image_files)

# Load and preprocess the selected image
if selected_image:
    image = Image.open(selected_image).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    st.image(image, caption="Selected Image", use_column_width=True)
    
    # Make a prediction
    prediction = model.predict(img_array)[0][0]
    st.write(f"Prediction: {'Human Detected' if prediction > 0.5 else 'No Human Detected'}")
    st.write(f"Confidence: {prediction:.2f}")

# Show model accuracy on validation data
loss, accuracy = model.evaluate(train_generator)
st.write(f"Model Validation Accuracy: {accuracy:.2f}")
