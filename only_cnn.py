import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import time



@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="potato_disease2.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()



st.warning("""
⚠️ **Disclaimer:** This tool is intended for **educational use only** and **not** for real-world agricultural decisions.  
Always consult an **agriculture expert** for professional guidance.
""")


st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload an image of a potato leaf to detect diseases.")
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])



class_names = ['Early Blight', 'Healthy', 'Late Blight']



@st.cache_data
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)




if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

    with st.spinner('🧠 Analyzing the image...'):
        time.sleep(1)  # Simulate processing time
        input_data = preprocess_image(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        predicted_class = np.argmax(output_data, axis=1)[0]
        confidence = float(np.max(output_data))
        st.subheader("📊 Class Probabilities ⬇️")
        for name, prob in zip(class_names, output_data[0]):
            st.write(f"{name}: {prob*100:.2f}%")
            st.progress(float(prob))

    st.success(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")