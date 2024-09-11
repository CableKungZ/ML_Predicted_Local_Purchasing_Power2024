import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model and columns
model = joblib.load('models/Random_Forest_Model.pkl')
columns = joblib.load('models/Random_Forest_Columns.pkl')

# Load the training data to create the scaler
# Replace with the path to your training data CSV file
df = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')

# Drop columns and handle missing values as done during model training
df = df.drop(['Rank', 'Country'], axis=1)
df = df.dropna()
df = df.drop_duplicates()

# Create a new scaler using the training data
scaler = StandardScaler()
X = df[columns]  # Features used for scaling
X_scaled = scaler.fit_transform(X)

st.title("Local Purchasing Power Prediction")

st.sidebar.header("Input Features")

def user_input_features():
    cost_of_living_index = st.sidebar.slider("Cost of Living Index", 0, 200, 80)
    rent_index = st.sidebar.slider("Rent Index", 0, 200, 50)
    cost_of_living_plus_rent_index = st.sidebar.slider("Cost of Living Plus Rent Index", 0, 200, 65)
    groceries_index = st.sidebar.slider("Groceries Index", 0, 200, 75)
    restaurant_price_index = st.sidebar.slider("Restaurant Price Index", 0, 200, 70)

    data = {
        'Cost of Living Index': cost_of_living_index,
        'Rent Index': rent_index,
        'Cost of Living Plus Rent Index': cost_of_living_plus_rent_index,
        'Groceries Index': groceries_index,
        'Restaurant Price Index': restaurant_price_index
    }
    features = np.array([list(data.values())])
    return features, data

input_features, input_data = user_input_features()

st.write("### Input Features")
st.write(input_data)

# Prepare the features for the model
input_features_df = pd.DataFrame(input_features, columns=columns)

# Scale the input features
input_features_scaled = scaler.transform(input_features_df)

# Predict the output using the loaded model
prediction = model.predict(input_features_scaled)

st.write(f"### Predicted Local Purchasing Power Index: **{prediction[0]:.2f}**")
