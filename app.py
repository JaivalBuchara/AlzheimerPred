import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = 'Osteoporosis_Model_binary.h5'
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Function to preprocess the image
def preprocess_image(img):
    # Resize to target size (244, 244) as done during training
    img = img.resize((244, 244))
    
    # Convert to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Streamlit app
st.title("Alzheimer's Disease Prediction")
st.write("Upload an MRI image to predict the stage of Alzheimer's Disease.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image.")

    # Preprocess image and make prediction
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)

    # Get prediction result
    predicted_class_index = np.argmax(predictions)
    result = class_labels[predicted_class_index]

    # Show prediction result
    st.write(f"Prediction: {result}")
