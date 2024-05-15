import streamlit as st
import tensorflow as tf
import cv2
import glob
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('yoga_pose_model.h5')
    return model

# Define the class names
class_names = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

# Function to preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Streamlit interface
st.title("Yoga Pose Classifier")
st.header("Identify Yoga Poses from Images")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = class_names[np.argmax(score)]
    st.write(f"The image is most likely a {result} pose with a {100 * np.max(score):.2f}% confidence.")
