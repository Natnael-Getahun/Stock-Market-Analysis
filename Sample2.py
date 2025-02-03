import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from workalendar.usa import UnitedStates
from scipy.stats.mstats import winsorize
import ta.momentum
import ta.volatility
import ta.trend
from dotenv import load_dotenv
import os
from scipy.stats import boxcox
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchmetrics
import streamlit as st
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objs as go
os.system("pip install -r requirements.txt")
import sys
import os
import pickle
from prophet import Prophet


# Standard library Import
import os
import warnings
import pickle


# Data Handling & Processing
import numpy as np
import pandas as pd
import yfinance as yf

# Machine Learning & Statistics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


# Time Series & Forecasting
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly

# Visualization
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

# Technical Analysis
import ta.momentum
import ta.trend
import ta.volatility

# Holiday Calendars
from workalendar.usa import UnitedStates

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import pickle
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



# Sample tickers
tickers = ["BRK-A", "BRK-B", "ORCL", "META", "KO", "IBM", "TSLA"]

# Load dataset
DATA_URL = "https://raw.githubusercontent.com/Natnael-Getahun/Stock-Market-Predictions/main/Datasets/BRK_A_medium.csv"
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
    return data

url_BRK_A_medium = load_data(DATA_URL)

# Set page configuration
st.set_page_config(page_title="Stock Market Prediction Dashboard", page_icon="🟢", layout="wide")
st.title("Stock Market Prediction Dashboard")


# Custom CSS to change the background to dark gray






st.markdown(
    """
    <style>
    .stApp {
        background-color: #2e2e2e;
        color: white;
    }
    .stSidebar {
        background-color: #2e2e2e;
        color: white;
    }
    .css-1d391kg {
        color: white;
    }
    .css-1d391kg a {
        color: #1f77b4;
    }
    .css-1d391kg a:hover {
        color: #ff7f0e;
    }
    </style>
    """,
    unsafe_allow_html=True
)











# input to selct stock and date range
stock_ticker = st.sidebar.selectbox('**Select a Stock:**', tickers, help="Choose the stock ticker you want to analyze.")
# Select start date
start_date = st.sidebar.date_input("Start Date",
    help="Select the start date for the data range.")

# Select end date
end_date = st.sidebar.date_input("##End Date",
    help="Select the end date for the data range.")

st.sidebar.markdown("#### Selected Stock and Date Range")
st.sidebar.info(f"**Stock Ticker:** {stock_ticker}")
st.sidebar.info(f"**Start Date:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.info(f"**End Date:** {end_date.strftime('%Y-%m-%d')}")


# Set up the sidebar with forecast range input box
forecast_range = st.sidebar.slider("Forecast Range (days)", min_value=30, max_value=365, value=90, step=30)
st.sidebar.write(f"Forecast Range: {forecast_range} days")


# Convert dates to datetime format with timezone
start_date = pd.Timestamp(start_date).tz_localize("UTC")
end_date = pd.Timestamp(end_date).tz_localize("UTC")


# Filter data
filtered_data = url_BRK_A_medium[(url_BRK_A_medium["Datetime"] >= start_date) & (url_BRK_A_medium["Datetime"] <= end_date)]


# Function to calculate evaluation metrics 
def calculate_metrics (y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape


# Placeholder for real-time data updates
placeholder = st.empty()

def fetch_live_data():
    return pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=17520, freq="h"),
        "close": np.random.randint(100, 500, 17520),
        "volume": np.random.randint(100000, 1000000, 17520)
    })

# Compute KPI metrics
Daily_Returns = round(url_BRK_A_medium["Close"].pct_change().mean() * 100, 4)
Volatility = url_BRK_A_medium["Close"].pct_change().std()
Trading_Volume = url_BRK_A_medium["Volume"].mean()

# Display KPI metrics
with placeholder.container():
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Daily Returns", value=f"{Daily_Returns}%", delta=f"{Daily_Returns:.2f}%")
    col2.metric(label="Volatility", value=f"{Volatility:.2%}", delta=f"{Volatility:.2%}")
    col3.metric(label="Trading Volume", value=f"{Trading_Volume:,.0f}", delta=f"{Trading_Volume:,.0f}")

# Charts
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("#### Closing Price Over Time")
    fig = px.line(url_BRK_A_medium, x="Datetime", y="Close")
    st.write(fig)
with fig_col2:
    st.markdown("#### Trading Volume Over Time")
    fig = px.histogram(url_BRK_A_medium, x="Datetime", y="Volume")
    st.write(fig)

st.markdown("#### Detailed Data View")
st.dataframe(url_BRK_A_medium)
time.sleep(0.5)

# Define the save_model function
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# A function to run prophet model
def run_prophet(
    data, 
    train = False,
    predict=True,
    prophet=True,
    cross_validate=False, 
    periods=50, 
    seasonality_mode='multiplicative', 
    changepoint_prior_scale=0.1, 
    seasonality_prior_scale=5.0, 
    n_changepoints=50, 
    yearly_seasonality=False, 
    weekly_seasonality=True, 
    daily_seasonality=True,
    initial='365 days',
    period='30 days', 
    horizon='90 days'
):
    """
    Train a Facebook Prophet model on stock market data, perform cross-validation, 
    and make future predictions.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing stock market data with 'Datetime', 'Close', 'Volume', 
        'MACD', and 'ATR' columns.
    train : bool, optional
        Whether to perform training on the model. Default is True.
    predict : bool, optional
        Whether to make future predictions. Default is True.
    prophet : a prophet model
        A model that can be used to make predictions with. Default is None as model can be trained from scratch.
    cross_validate : bool, optional
        Whether to perform cross-validation on the model. Default is True.
    periods : int, optional
        Number of future periods to predict. Default is 50.
    seasonality_mode : str, optional
        Prophet's seasonality mode ('additive' or 'multiplicative'). Default is 'multiplicative'.
    changepoint_prior_scale : float, optional
        Flexibility of the trend change points. Default is 0.1.
    seasonality_prior_scale : float, optional
        Strength of seasonality prior. Default is 5.0.
    n_changepoints : int, optional
        Number of trend changepoints. Default is 50.
    yearly_seasonality : bool, optional
        Enable or disable yearly seasonality. Default is False.
    weekly_seasonality : bool, optional
        Enable or disable weekly seasonality. Default is True.
    daily_seasonality : bool, optional
        Enable or disable daily seasonality. Default is True.
    initial : str, optional
        Initial training period for cross-validation. Default is '365 days'.
    period : str, optional
        Period between cutoff dates for cross-validation. Default is '30 days'.
    horizon : str, optional
        Forecast horizon for cross-validation. Default is '90 days'.

    Returns:
    --------
    prophet : Prophet
        Trained Prophet model.
    transformed_predictions : pd.DataFrame
        DataFrame containing log-transformed future predictions.
    predictions : pd.Series
        Actual predictions with logarithm reversed.
    """
    
    # Prepare the dataset
    df = data[['Close', 'Volume', 'MACD', 'ATR']].reset_index()
    df.rename(columns={'Datetime': 'ds', 'Date' : 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone information
    df['y'] = np.log(df['y'])  # Apply log transformation for better trend modeling

    if train:
        # Create a holiday effect based on US Federal holidays
        calendar = UnitedStates()
        holidays = []
        for year in range(data.index.min().year, data.index.max().year + 2):
            holidays.extend(calendar.holidays(year))
        
        # Convert holiday names to holiday dates
        holiday_dates = [holiday[0] for holiday in holidays]  # Extract dates from holiday objects
        holidays_df = pd.DataFrame({
            'holiday': 'earnings_release',
            'ds': pd.to_datetime(holiday_dates),
            'lower_window': -5,  # Effect starts 5 days before
            'upper_window': 5,   # Effect ends 5 days after
        })
        
        # Initialize Prophet model with provided parameters
        prophet = Prophet(
            seasonality_mode=seasonality_mode, 
            changepoint_prior_scale=changepoint_prior_scale, 
            seasonality_prior_scale=seasonality_prior_scale, 
            holidays=holidays_df,
            n_changepoints=n_changepoints, 
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add additional regressors
        prophet.add_regressor('Volume')
        prophet.add_regressor('MACD')
        prophet.add_regressor('ATR')
        
        # Train the model
        prophet.fit(df)
        print("Prophet model fitted.\n")
    
    # Perform cross-validation
    if cross_validate:
        print("Starting cross-validation...\n")
        df_cv = cross_validation(prophet, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        print(df_p.head())
        print("Cross-validation finished.\n")
    
    # Prediction phase
    if predict:
        print("Starting prediction...\n")
        
        def predict_regressor(data, regressor, periods):
            """
            Predicts future values of a given regressor using Prophet.

            Parameters:
            - data (pd.DataFrame): Historical dataset containing 'ds' and the specified regressor.
            - regressor (str): Name of the regressor column.
            - periods (int): Number of periods to predict into the future.

            Returns:
            - pd.DataFrame: DataFrame with predicted values for the specified regressor.
            """
            
            model = Prophet(
                seasonality_mode='multiplicative', 
                changepoint_prior_scale=0.1, 
                seasonality_prior_scale=5.0, 
                holidays=holidays_df,
                n_changepoints=50, 
                yearly_seasonality=False, 
                weekly_seasonality=True, 
                daily_seasonality=True
            )
            
            df_reg = data[['ds', regressor]].rename(columns={regressor: 'y'})
            model.fit(df_reg)
            
            future = model.make_future_dataframe(periods=periods)
            future[regressor] = data[regressor]
            
            prediction = model.predict(future)
            return prediction
        
        # Create future dataframe for prediction
        future = prophet.make_future_dataframe(periods=periods)
        
        # Predict each regressor separately
        future['Volume'] = predict_regressor(df, 'Volume', periods)['yhat']
        future['MACD'] = predict_regressor(df, 'MACD', periods)['yhat']
        future['ATR'] = predict_regressor(df, 'ATR', periods)['yhat']
        
        # Generate final predictions
        transformed_predictions = prophet.predict(future)
        predictions = np.exp(transformed_predictions["yhat"])
        print("Finished prediction.")
    
    return prophet, transformed_predictions, predictions


# A function to save the prophet model
def save_prophet_model(model, model_name):
    """
    Save a trained Prophet model to a specified directory.

    Parameters:
    model (Prophet): The trained Prophet model to be saved.
    model_name (str): The name to be used for the saved model file (without file extension).

    The model will be saved as a .pkl file in the 'models' directory.
    If the 'models' directory doesn't exist, it will be created.

    Example:
    save_prophet_model(model, 'stock_prediction_model')
    """
    # Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Define the path to save the model
    model_path = os.path.join('models', f'{model_name}.pkl')
    
    # Save the Prophet model to the specified path
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {model_path}")


# Prophet model function
def run_prophet(data, periods=365):
    data = data.rename(columns={"Datetime": "ds", "Close": "y"})
    data['ds'] = data['ds'].dt.tz_localize(None)  # Remove timezone information
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Save Prophet and run data prediction model
if st.button("Run Stock Prediction Model"):
    prophet_model, forecast = run_prophet(url_BRK_A_medium)
    save_model(prophet_model, "BRK_A_mestreamdium")
    
    # Create a plotly figure with lower and upper bounds
    fig = px.line()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines',
        name='Predicted Closing Price'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0.2)')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(0,255,0,0.2)')
    ))
    
    fig.update_layout(title='Predicted Stock Closing')
    st.plotly_chart(fig)
    # Calculate evaluation metrics
    y_true = url_BRK_A_medium['Close'][-forecast_range:]
    y_pred = forecast['yhat'][-forecast_range:]
    rmse, mae, mape = calculate_metrics(y_true, y_pred)
    
    # Display evaluation metrics
    st.markdown("### Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<p style='color:black;'>RMSE: <span style='color:green;'>{rmse:.2f}</span></p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='color:black;'>MAE: <span style='color:green;'>{mae:.2f}</span></p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p style='color:black;'>MAPE: <span style='color:green;'>{mape:.2%}</span></p>", unsafe_allow_html=True)

    #fig = px.line(forecast, x='ds', y='yhat', title='Predicted Closing Price')
    #st.plotly_chart(fig)
    st.write("Detail Predictions:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])


