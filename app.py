import streamlit as st
import joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('models/agg.pkl')

# สร้างอินเตอร์เฟซ Streamlit
st.title('Cost of Living Prediction')

# สร้างช่องรับอินพุตสำหรับดัชนีต่าง ๆ
cost_of_living_index = st.number_input('Cost of Living Index', min_value=0.0, max_value=1000.0, value=50.0)
rent_index = st.number_input('Rent Index', min_value=0.0, max_value=1000.0, value=50.0)
groceries_index = st.number_input('Groceries Index', min_value=0.0, max_value=1000.0, value=50.0)
restaurant_price_index = st.number_input('Restaurant Price Index', min_value=0.0, max_value=1000.0, value=50.0)
local_purchasing_power_index = st.number_input('Local Purchasing Power Index', min_value=0.0, max_value=1000.0, value=50.0)

# เมื่อผู้ใช้กดปุ่มทำนาย
if st.button('Predict'):
    # เตรียมข้อมูลอินพุต
    input_data = np.array([[cost_of_living_index, rent_index, groceries_index, restaurant_price_index, local_purchasing_power_index]])
    
    # ใช้โมเดลในการทำนายผล
    result = model.predict(input_data)
    
    # แสดงผลลัพธ์การทำนาย
    st.write('Prediction result:', result[0])
