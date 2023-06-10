import streamlit as st
import requests
import tensorflow as tf

# Specify the URL of the model folder on Google Drive
folder_url = "https://drive.google.com/drive/folders/1lPhuBWhJ5nF8118nRPV3X-vADUkelAp5?usp=drive_link"

# Download the folder metadata
response = requests.get(folder_url)
folder_contents = response.content.decode('utf-8')

# Extract the file IDs from the folder metadata
file_ids = []
for line in folder_contents.split("\n"):
    if "data-id" in line:
        file_id = line.split('data-id="')[1].split('"')[0]
        file_ids.append(file_id)

# Load the model
with st.spinner('Loading the model...'):
    for file_id in file_ids:
        file_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(file_url)
        # Process the downloaded file
        # Load the model using the appropriate code for your model type
        model = tf.keras.models.load_model(response.content)

st.success('Model loaded successfully!')

# Use the loaded model for your Streamlit app

