# Standard Library Imports
import os
import sys
import warnings
import pickle

# Data Handling & Processing
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox

# Machine Learning & Statistics
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchmetrics
import tensorflow as tf  # Added TensorFlow import
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Time Series & Forecasting
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import mplfinance as mpf
import time
import time

# Technical Analysis
import ta.momentum
import ta.volatility
import ta.trend

# Holiday Calendars
from workalendar.usa import UnitedStates

# Environment Variables
from dotenv import load_dotenv

# Interactive Widgets
import ipywidgets as widgets
from IPython.display import display

# Streamlit
import streamlit as st

# Add the directory containing stock_prediction_function.py to the Python path
sys.path.append(os.path.dirname(__file__))

# Import prediction functions from stock_prediction_function.py
import stock_prediction_functions as spf
from stock_prediction_functions import get_data, general_description, get_plots, clean_data, summary


# Sample tickers
tickers = ["BRK-A", "BRK-B", "ORCL", "META", "KO", "IBM", "TSLA"]

# Load dataset
DATA_URL = "https://raw.githubusercontent.com/Natnael-Getahun/Stock-Market-Predictions/main/Datasets/BRK_A_medium.csv"

#Define the 
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
    return data 

url_BRK_A_medium = load_data(DATA_URL)

#a function to clean the live data 


# Set page configuration
st.set_page_config(page_title="Stock Market Prediction Dashboard", page_icon="🟢", layout="wide")
st.title("Stock Market Prediction Dashboard")


# Custom CSS to change the background to dark gray and sliders/radio buttons to blue


# input to selct stock and date range
stock_ticker = st.sidebar.selectbox ('**Select a Stock:**', options = tickers, index = 0, help="Choose the stock ticker you want to analyze.", format_func=lambda x: f"📈 {x}")


# Function to get data intervals
def get_data_intervals(ticker):
    short_range = get_data(ticker, period='1mo', interval='2m')
    medium_range = get_data(ticker, period='2y', interval='1h')
    long_range = get_data(ticker, period='max', interval='1d')
    return short_range, medium_range, long_range


#get_data_intervals function
short_range, medium_range, long_range = get_data_intervals(stock_ticker)

# Clean the data
short_range = clean_data(short_range)
medium_range = clean_data(medium_range)
long_range = clean_data(long_range)

# load and run feature engineering function
def feature_engineering_and_save_data(data, filename):
    # Feature Engineering
    data = spf.feature_engineering(data)
    # Save the data
    data.to_csv(filename, index=False)
    return data


# Dropdown to select data interval "short", "medium", "long"
interval_option = st.sidebar.selectbox(
    "**Select Data Interval:**",
    ("Short Range", "Medium Range", "Long Range"),
    help="Short Range: 1 month of data with a 2-minute interval. \n"
         "Medium Range: 2 years of data with a 1-hour interval. \n"
         "Long Range: Maximum available data with a 1-day interval"
)

# Display selected data interval
st.markdown("### Data Interval")
if interval_option == "Short Range":
    st.write("**Short Range:** 1 month of data with a 2-minute interval.")
    st.dataframe(short_range)
elif interval_option == "Medium Range":
    st.write("**Medium Range:** 2 years of data with a 1-hour interval.")
    st.dataframe(medium_range)
elif interval_option == "Long Range":
    st.write("**Long Range:** Maximum available data with a 1-day interval.")
    st.dataframe(long_range)

# Select start date 
start_date = st. sidebar.date_input("Start Date",
    help="Select the start date for the data range.")

# Select end date
end_date = st.sidebar.date_input("##End Date",
    help="Select the end date for the data range.")

st.sidebar.markdown("#### Selected Stock and Date Range")
st.sidebar.info(f"**Stock Ticker:** {stock_ticker}")
st.sidebar.info(f"**Start Date:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.info(f"**End Date:** {end_date.strftime('%Y-%m-%d')}")


# Set up the sidebar with forecast range input box
forecast_range = st.sidebar.slider("Forecast Range (hours)", min_value=1, max_value=50, value=6, step=1)
st.sidebar.write(f"Forecast Range: {forecast_range} hours")

#praphet parameters on sidebbar


# Sidebar options for seasonality
st.sidebar.markdown("### Seasonality Options")
yearly_seasonality = st.sidebar.radio("Yearly Seasonality", ["On", "Off"], index=1)
weekly_seasonality = st.sidebar.radio("Weekly Seasonality", ["On", "Off"], index=0)
daily_seasonality = st.sidebar.radio("Daily Seasonality", ["On", "Off"], index=0)

# Map "On" and "Off" to True and False
yearly_seasonality = True if yearly_seasonality == "On" else False
weekly_seasonality = True if weekly_seasonality == "On" else False
daily_seasonality = True if daily_seasonality == "On" else False


# Convert dates to datetime format with timezone
start_date = pd.Timestamp(start_date).tz_localize("UTC")
end_date = pd.Timestamp(end_date).tz_localize("UTC")


# Filter date
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


# Prophet model function
def run_prophet(data, periods=2, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto'):
    data = data.rename(columns={"Datetime": "ds", "Close": "y"})
    data['ds'] = data['ds'].dt.tz_localize(None)  # Remove timezone information
    model = Prophet(yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Save Prophet and run data prediction model
if st.button("Run Stock Prediction Model"):
    prophet_model, forecast = run_prophet(url_BRK_A_medium, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
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


























# Perform feature engineering for the selected data interval
feature_engineered_data = spf.feature_engineering_and_save_data(cleaned_data, f"{stock_ticker}_{data_interval.lower().replace(' ', '_')}")

# Using the feature_engineered_data to perform the prophet model for the selected stock
Prophet_model, forecast = spf.run_prophet(feature_engineered_data, periods=forecast_range, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)

