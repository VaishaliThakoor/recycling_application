import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

# Load pre-trained model
model = tf.keras.applications.MobileNetV2()

# Define labels
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", 
                                      "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
with open(labels_path) as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

st.title("Image Classifier")

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    predicted_class = labels[predicted_label]

    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")



