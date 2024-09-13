import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler parameters
model = joblib.load('models/Linear_Regression_Model.pkl')

scaler_params = joblib.load('models/Scaler_Parameters.pkl')

scaler = StandardScaler()
scaler.mean_ = scaler_params['mean']
scaler.scale_ = scaler_params['scale']

# Define the Streamlit app
st.title('Rent Index Prediction')

# Create input fields for user to enter values
cost_of_living_index = st.number_input('Cost of Living Index', min_value=0.0, value=101.1)
groceries_index = st.number_input('Groceries Index', min_value=0.0, value=109.1)
restaurant_price_index = st.number_input('Restaurant Price Index', min_value=0.0, value=97.0)
local_purchasing_power_index = st.number_input('Local Purchasing Power Index', min_value=0.0, value=158.7)

# Predict button
if st.button('Predict'):
    # Prepare the input data
    new_data = pd.DataFrame({
        'Cost of Living Index': [cost_of_living_index],
        'Groceries Index': [groceries_index],
        'Restaurant Price Index': [restaurant_price_index],
        'Local Purchasing Power Index': [local_purchasing_power_index]
    })

    # Standardize the input data
    new_data_scaled = scaler.transform(new_data)

    # Predict
    predicted_rent_index = model.predict(new_data_scaled)

    # Display the result
    st.write(f"Predicted Rent Index: {predicted_rent_index[0]:.2f}")
    st.write(f"Rent Index Cheaper than Rent in New York City ({100-predicted_rent_index[0]:.2f} %)")
