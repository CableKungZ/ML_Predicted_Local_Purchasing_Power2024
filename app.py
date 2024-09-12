import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score

# Load the saved model
agg = joblib.load('agg.pkl')

# Title of the Streamlit app
st.title('Hierarchical Clustering Model Deployment')

# Sidebar inputs for each feature
st.sidebar.header('Input Features')
cost_of_living_index = st.sidebar.number_input('Cost of Living Index', min_value=0.0, step=0.1)
rent_index = st.sidebar.number_input('Rent Index', min_value=0.0, step=0.1)
groceries_index = st.sidebar.number_input('Groceries Index', min_value=0.0, step=0.1)
restaurant_price_index = st.sidebar.number_input('Restaurant Price Index', min_value=0.0, step=0.1)
local_purchasing_power_index = st.sidebar.number_input('Local Purchasing Power Index', min_value=0.0, step=0.1)

# Create a DataFrame from the input
input_data = {
    'Cost of Living Index': [cost_of_living_index],
    'Rent Index': [rent_index],
    'Groceries Index': [groceries_index],
    'Restaurant Price Index': [restaurant_price_index],
    'Local Purchasing Power Index': [local_purchasing_power_index]
}
df = pd.DataFrame(input_data)

# Display the input DataFrame
st.write('### Input Data')
st.write(df)

# Predict clusters using the loaded model
df['cluster'] = agg.fit_predict(df)

# Display the results
st.write('### Clustered Data')
st.write(df)

# Calculate and display silhouette score
if len(df) > 1:  # Ensure more than one sample for silhouette score calculation
    silhouette = silhouette_score(df.drop('cluster', axis=1), df['cluster'])
    st.write(f'### Silhouette Score: {silhouette:.4f}')
else:
    st.write('### Silhouette Score: Not applicable for a single input.')
