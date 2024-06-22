# Electricity Consumption Analysis

This project is a Streamlit application for analyzing and visualizing electricity consumption data. Users can upload their own CSV files to analyze the trends in electricity consumption or download a sample file to get started. The application provides various visualization options and the ability to train a machine learning model (XGBoost) to make predictions based on the data.

## Features

- **File Upload**: Users can upload a CSV file containing electricity consumption data.
- **Sample Data**: A sample CSV file is available for download for users who don't have their own data. (There are 2 csv files in my repository that can be downloaded)
- **Time Series Plot**: Visualize the electricity consumption over time.
- **Feature Creation**: Generate additional time-based features for analysis.
- **Hourly and Monthly Consumption Plots**: Boxplots showing the distribution of electricity consumption by hour and by month.
- **XGBoost Model Training**: Train an XGBoost model on the uploaded data to make predictions and evaluate its performance.
- **Prediction Plot**: Visualize the true data against the model's predictions.

## Requirements

- streamlit
- pandas
- matplotlib
- seaborn
- plotly
- xgboost
- scikit-learn

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
