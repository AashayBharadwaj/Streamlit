import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path, date_col_index=0):
    df = pd.read_csv(file_path, parse_dates=[date_col_index])
    datetime_column = df.columns[date_col_index]  # Automatically gets the datetime column name
    df = df.set_index(datetime_column)
    return df

def plot_time_series(df):
    color_pal = sns.color_palette()
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df, label='Electricity Consumption', color=color_pal[0])
    plt.title('Energy Use in MW')
    plt.legend()
    plt.xlabel('Datetime')
    plt.ylabel('Consumption')
    st.pyplot(plt.gcf())  # Show the plot within Streamlit

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def plot_hourly_consumption_new(df):
    second_column = df.columns[0]  # Dynamically get the second column's header
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='hour', y=second_column)  # Use the second column for the y-axis
    ax.set_title(f'{second_column} by Hour')  # Dynamic title based on the column name
    st.pyplot(fig)

def plot_monthly_consumption_new(df):
    second_column = df.columns[0]  # Dynamically get the second column's header
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='month', y=second_column)
    ax.set_title('MW by month new')
    st.pyplot(fig)

def train_xgboost(train, test, df):
    FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    TARGET = train.columns[0]  # Assume the second column in the DataFrame is the target
    
    X_train = train[FEATURES]
    Y_train = train[TARGET]
    X_test = test[FEATURES]
    Y_test = test[TARGET]
    
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R^2 Score: {r2}")
    
    test['prediction'] = predictions
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    
    return df

def plot_predictions(df, target):
    fig, ax = plt.subplots(figsize=(15, 5))
    df[[target]].plot(ax=ax)
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    st.pyplot(fig)

def main():
    st.title("Electricity Consumption Analysis")
    uploaded_file = st.file_uploader("Upload your data file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.write("Visualization of Energy Use:")
        plot_time_series(df)

        df = create_features(df)

        # Split the data into train and test sets
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        
        # Let the user choose the plot type
        plot_type = st.selectbox(
            "Choose the type of consumption plot:",
            ["Hourly Consumption", "Monthly Consumption", "Train XGBoost Model"],
            index=0  # Default selection
        )

        # Display the chosen plot
        if plot_type == "Hourly Consumption":
            st.write("Hourly Consumption Plot:")
            plot_hourly_consumption_new(df)
        elif plot_type == "Monthly Consumption":
            st.write("Monthly Consumption Plot:")
            plot_monthly_consumption_new(df)
        elif plot_type == "Train XGBoost Model":
            st.write("Training XGBoost Model and Making Predictions:")
            df = train_xgboost(train, test, df)
            plot_predictions(df, train.columns[0])

if __name__ == "__main__":
    main()
