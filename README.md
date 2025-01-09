# Machine Learning for Rent Index Prediction

This project focuses on predicting the **Rent Index** using machine learning techniques, leveraging data on cost of living indices across various countries. The model is deployed as a Streamlit web application, enabling users to interact with the prediction system easily.

## Dataset Source

The dataset for this project is sourced from Kaggle:

[Cost of Living Index by Country (2024)](https://www.kaggle.com/datasets/myrios/cost-of-living-index-by-country-by-number-2024)

### About the Dataset

The dataset provides cost of living indices for various countries, benchmarked against New York City (NYC) as the baseline. NYC is assigned an index value of 100%, and all other indices are relative to it. The indices include:

1. **Cost of Living Index (Excl. Rent):** Prices of goods and services, excluding rent.
2. **Rent Index:** Average rental prices compared to NYC.
3. **Cost of Living Plus Rent Index:** Combines consumer goods and rent prices.
4. **Groceries Index:** Grocery prices relative to NYC.
5. **Restaurant Price Index:** Prices of dining out compared to NYC.
6. **Local Purchasing Power Index:** Relative purchasing power based on average net salary.

For detailed information, visit: [Numbeo Explanation](https://www.numbeo.com/cost-of-living/cpi_explained.jsp)

---

### Example Indices

| **Metric**                  | **Example Value** |
|-----------------------------|--------------------|
| Cost of Living Index        | 101.10            |
| Groceries Index             | 109.10            |
| Restaurant Price Index      | 97.00             |
| Local Purchasing Power Index| 158.70            |

## Deployment

The Rent Index Prediction model is deployed via Streamlit. You can access the application here:

[Streamlit Rent Index Prediction App](https://ml-costoflivingindexbycountry.streamlit.app/)

## Features

- **Data Insights:** Provides visualizations and analysis of cost of living indices.
- **Predictive Model:** Predicts the Rent Index using machine learning based on other indices.
- **User-Friendly:** Easy-to-use interface for inputting data and receiving predictions.

## Model Selection

For this project, various machine learning models were evaluated to determine the most suitable one for predicting the Rent Index. Models were assessed based on their accuracy, performance metrics, and alignment with the dataset's characteristics. The final model was chosen as the one with the best and most reliable results, ensuring a balance between accuracy and interpretability.

## How to Use

1. Access the deployed application via the link above.
2. Input values for various indices such as Cost of Living, Groceries, Restaurant Price, and Local Purchasing Power.
3. View the predicted Rent Index.

## Technologies Used

- Python
- Scikit-learn for building machine learning models
- Pandas and NumPy for data processing
- Streamlit for web application deployment
- Matplotlib and Seaborn for data visualization
- ML supervise MODEL

## Contribution

Contributions are welcome! Feel free to fork this repository and submit your enhancements through a pull request.


---

Thank you for exploring this project. Your feedback and contributions are highly valued!
