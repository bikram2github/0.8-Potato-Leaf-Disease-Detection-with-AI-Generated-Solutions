#checking .keras model

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import time


@st.cache_resource
def load_trained_model():
    return load_model("potato_disease.keras")

model = load_trained_model()


st.warning("⚠️ This application is for educational purposes only and should not be used as a substitute for professional agricultural advice.")
st.title("Potato Disease Detection")    
st.write("Upload an image of a potato leaf to detect diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
class_names = ['Early Blight', 'Healthy', 'Late Blight']
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

    with st.spinner('Processing...'):
        time.sleep(1)  # Simulate processing time
        input_data = preprocess_image(image)
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        st.subheader("Class Probabilities")
    for name, prob in zip(class_names, predictions[0]):
        st.write(f"{name}: {prob*100:.2f}%")
        st.progress(float(prob))



    st.success(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")