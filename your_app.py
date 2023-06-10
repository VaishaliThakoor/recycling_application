import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2()

# Define the labels for ImageNet classes
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = tf.keras.utils.get_file("ImageNetLabels.json", LABELS_URL)

# Load the ImageNet labels
with open(labels) as f:
    imagenet_labels = np.array(f.read().splitlines())

# Streamlit app code
st.title("Image Classification with TensorFlow")

# Upload and classify an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    resized_image = image.resize((224, 224))
    normalized_image = np.array(resized_image) / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    # Classify the image
    predictions = model.predict(input_image)
    predicted_label = imagenet_labels[np.argmax(predictions)]

    # Display the predicted label
    st.subheader("Prediction:")
    st.write(predicted_label)
