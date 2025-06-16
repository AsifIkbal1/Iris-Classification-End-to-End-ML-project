# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 🔹 Model এবং Scaler লোড করা
with open('/Users/apple/Downloads/Project/Iris-Classification-End-to-End-ML-project/dataset/svc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('/Users/apple/Downloads/Project/Iris-Classification-End-to-End-ML-project/dataset/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 🔹 Species mapping
species_map = {
    0: 'Setosa 🌸',
    1: 'Versicolour 🌿',
    2: 'Virginica 🌺'
}

# 🔹 Streamlit UI
st.title("🌼 Iris Flower Species Prediction App")

st.write("নিচের মান গুলো ইনপুট দাও এবং আমরা বলে দিব তোমার ফুলটি কোন প্রজাতির।")

# 🔹 Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# 🔹 Feature তৈরি
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
scaled_input = scaler.transform(input_data)

# 🔹 Prediction
if st.button("📍 Predict"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🌟 Predicted Species: **{species_map[prediction]}**")

# 🔹 Optional: Show dataset
with st.expander("📄 Show Dataset"):
    df = pd.read_csv("/Users/apple/Downloads/Project/Iris-Classification-End-to-End-ML-project/dataset/iris.csv")
    st.dataframe(df)
