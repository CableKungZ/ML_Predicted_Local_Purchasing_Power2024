import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and label encoder
model = joblib.load('models/Random_Forest_Model.pkl')
# Load any encoder if used for categorical features
# encoder = joblib.load('models/RF_Label_Encoder.pkl') 
columns = joblib.load('models/Random_Forest_Columns.pkl.pkl')

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

# Predict the output using the loaded model
prediction = model.predict(input_features_df)

# If you have categorical predictions, you might need to decode them
# prediction_label = encoder.inverse_transform(prediction)

st.write(f"### Predicted Local Purchasing Power Index: **{prediction[0]:.2f}**")
