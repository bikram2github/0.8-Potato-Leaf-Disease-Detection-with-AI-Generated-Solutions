# 0.8-Potato-Leaf-Disease-Detection-with-AI-Generated-Solutions
Potato Leaf Disease Detection with AI-Generated Solutions

Web Link : https://potato-leaf-disease-detection-with-ai-solutions-bikram.streamlit.app/
 
# 🥔 Potato Leaf Disease Detection using CNN & GenAI

An end-to-end AI-powered web application that detects potato leaf diseases from images and generates intelligent explanations and treatment recommendations using **Deep Learning** and **Large Language Models (LLMs)**.

---

## 📌 Project Overview

Potato crops are highly vulnerable to diseases such as **Early Blight** and **Late Blight**, which can significantly reduce crop yield if not identified early.  
This project leverages a **Convolutional Neural Network (CNN)** for image classification and integrates **Generative AI** to provide actionable disease insights.

The system classifies potato leaf images into:
- 🌿 **Healthy**
- 🍂 **Early Blight**
- 🍁 **Late Blight**

and delivers AI-generated explanations and remedies through an interactive web application.

---

## 🚀 Key Features

- CNN-based image classification using **TensorFlow**
- Transfer Learning with **ResNet50** and **EfficientNetB0**
- Image preprocessing and augmentation for better generalization
- Achieved **~98% validation accuracy**
- Model evaluation using:
  - Confusion Matrix
  - Classification Report
- LLM-powered disease explanation and treatment recommendations
- Interactive **Streamlit** web application
- End-to-end pipeline: **Image → Prediction → AI Insights**

---

## 🧠 Tech Stack

**Programming:** Python  
**Deep Learning:** TensorFlow, Keras, CNN  
**Models:** ResNet50, EfficientNetB0  
**Generative AI:** LangChain, Groq LLM  
**Web Framework:** Streamlit  
**Libraries:** NumPy, Pandas, Matplotlib  
**Tools:** Git, GitHub, Google Colab, VS Code  

---

## 🏗️ Workflow

1. Upload potato leaf image  
2. Image preprocessing and augmentation  
3. CNN-based disease classification  
4. LLM-generated disease explanation and remedies  
5. Results displayed via Streamlit web interface  

---

## ▶️ How to Run the Application

```bash
git clone <repository-url>
cd Potato-Leaf-Disease-Detection-with-AI-Generated-Solutions
pip install -r requirements.txt
streamlit run app.py
