import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('waste_classification_model.h5')
 
# Define the recycling categories
recycling_categories = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Create the Streamlit web app
def main():
    st.title("Recycling App")

    # Add an image uploader for the user to upload waste images
    uploaded_image = st.file_uploader("Upload an image of the waste")

    if uploaded_image is not None:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_image)
        processed_image = preprocess_image(image)

        # Make predictions using the trained model
        predictions = model.predict(processed_image)
        predicted_class = recycling_categories[np.argmax(predictions)]

        # Display the predicted recycling category
        st.subheader("Predicted Recycling Category:")
        st.write(predicted_class)

# Preprocess the image (resize, normalize, etc.) for model input
def preprocess_image(image):
    # Apply any necessary preprocessing steps (resize, normalize, etc.)
    processed_image = image.resize((128, 128))  # Example resizing to 224x224
    processed_image = np.array(processed_image) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    # Return the processed image
    return processed_image

if __name__ == "__main__":
    main()
