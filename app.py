import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv() 

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=api_key,
                    model="openai/gpt-oss-20b",
                    temperature=0.7,
                )

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

st.title("🥔 Potato Leaf Disease Detection with AI-Generated Solutions")
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
    st.image(image, caption='Uploaded Leaf Image', width='content')

    with st.spinner('🧠 Analyzing the image...'):
        time.sleep(1)  # Simulate processing time
        input_data = preprocess_image(image)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        predicted_class = np.argmax(output_data, axis=1)[0]
        disease_name = class_names[predicted_class]
        confidence = float(np.max(output_data))
        st.subheader("📊 Class Probabilities ⬇️")
        for name, prob in zip(class_names, output_data[0]):
            st.write(f"{name}: {prob*100:.2f}%")
            st.progress(float(prob))

    st.subheader(f"✅ Prediction: {class_names[predicted_class]}")     
    st.divider() 
    space = st.empty()
    space.write("")



    if st.button("🌱 Get Disease Information & Treatment Suggestions"):
        if disease_name == "Healthy":
            st.success("This potato leaf is healthy.")
            st.balloons()
        else:
            with st.spinner("Extracting information..."):
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system","You are an agricultural AI assistant specializing in potato leaf diseases. "
                            "Given the disease name, provide a clear, well-structured explanation that includes: "
                            "1. Cause, 2. Symptoms, and 3. Treatment or prevention methods. "
                            "Keep the language simple and easy to understand for farmers or students. "
                            "Write the answer attractively with short paragraphs and bullet points where suitable, "
                            "and keep the response within 300 words."),
                        ("user","The potato leaf is diagnosed with {disease_name}.")
                    ]
                )

                parser = StrOutputParser()               
                chain=prompt | llm | parser

                response = chain.invoke({"disease_name": disease_name})
                st.subheader("🌿 Disease Information & Treatment Suggestions")
                st.write(response)