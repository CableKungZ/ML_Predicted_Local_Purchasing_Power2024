import streamlit as st
import pandas as pd
import joblib
import numpy as np

# โหลดโมเดลและข้อมูลคอลัมน์
model = joblib.load('models/Logistic_Regression_Model.pkl')
columns = joblib.load('models/Logistic_Regression_Columns.pkl')

# ฟังก์ชันสำหรับการคาดการณ์
def predict_rent(features):
    # แปลงข้อมูลเป็น DataFrame
    input_data = pd.DataFrame([features], columns=columns)
    # ทำนายผล
    prediction = model.predict(input_data)
    return prediction[0]

# สร้างแอพ Streamlit
st.title('Rent Index Prediction')

# สร้างฟอร์มสำหรับการกรอกข้อมูล
st.sidebar.header('Input Features')
inputs = {}
for column in columns:
    inputs[column] = st.sidebar.number_input(f'{column}', value=0.0)

if st.sidebar.button('Predict'):
    features = [inputs[col] for col in columns]
    prediction = predict_rent(features)
    st.write(f'The predicted Rent Index is: {prediction:.2f}')
