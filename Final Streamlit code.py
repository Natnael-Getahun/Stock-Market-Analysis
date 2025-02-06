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
import time

# Add the directory containing stock_prediction_function.py to the Python path
sys.path.append(os.path.dirname(__file__))

# Import prediction functions from stock_prediction_function.py
import stock_prediction_functions as spf
from stock_prediction_functions import get_data, general_description, get_plots, clean_data, summary

# Suppress all warnings
warnings.filterwarnings('ignore')

# Sample tickers
tickers = ["BRK-A", "BRK-B", "ORCL", "META", "KO", "IBM", "TSLA"]


# Set page configuration
st.set_page_config(page_title="Stock Market Prediction Dashboard", page_icon="🟢", layout="wide")
st.title("Stock Market Prediction Dashboard")


# Sidebar for user input to select stock and date range
stock_ticker = st.sidebar.selectbox('Select a Stock:', options=tickers, index=0, help="Choose the stock ticker you want to analyze.")
data_interval = st.sidebar.selectbox('Select Data Interval:', options=["Short Range", "Medium Range", "Long Range"], help="Choose the data interval.")
start_date = st.sidebar.date_input("Start Date", help="Select the start date for the data range.")
end_date = st.sidebar.date_input("End Date", help="Select the end date for the data range.")

# display selected stock ticker, start date, end date, and data interval
st.sidebar.markdown("### Selected Stock and Date Range")
st.sidebar.info(f"**Stock Ticker:** {stock_ticker}")
st.sidebar.info(f"**Start Date:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.info(f"**End Date:** {end_date.strftime('%Y-%m-%d')}")
st.sidebar.info(f"**Data Interval:** {data_interval}")


# Fetch data based on user selection
if data_interval == "Short Range":
    data = spf.get_data(stock_ticker, period='1mo', interval='2m')
elif data_interval == "Medium Range":
    data = spf.get_data(stock_ticker, period='2y', interval='1h')
elif data_interval == "Long Range":
    data = spf.get_data(stock_ticker, period='max', interval='1d')
else:
    st.warning("Invalid data interval selected.")

# Clean the data
cleaned_data = spf.clean_data(data)

# Convert start_date and end_date to timezone-aware datetime
start_date = pd.to_datetime(start_date).tz_localize('UTC')
end_date = pd.to_datetime(end_date).tz_localize('UTC')


# Filter data based on selected date range
filtered_data = cleaned_data[(cleaned_data.index >= start_date) & (cleaned_data.index <= end_date)]



# Function to calculate evaluation metrics 
def calculate_metrics (y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape
   

# Placeholder for real-time data updates
placeholder = st.empty()

# Compute KPI metrics
Daily_Returns = round(cleaned_data[f"{stock_ticker}_Close"].pct_change().mean() * 100, 4)
Volatility = cleaned_data[f"{stock_ticker}_Close"].pct_change().std()
Trading_Volume = cleaned_data[f"{stock_ticker}_Volume"].mean()


# Display KPI metrics
with placeholder.container():
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Daily Returns", value=f"{Daily_Returns}%", delta=f"{Daily_Returns:.2f}%")
    col2.metric(label="Volatility", value=f"{Volatility:.2%}", delta=f"{Volatility:.2%}")
    col3.metric(label="Trading Volume", value=f"{Trading_Volume:,.0f}", delta=f"{Trading_Volume:,.0f}")

# Title for the charts
st.write(f"### {stock_ticker} Live Data")

# Create a two-column layout
col1, col2 = st.columns(2)

# Display the closing price plot in the first column
with col1:
    fig1 = px.line(cleaned_data, x=cleaned_data.index, y=f"{stock_ticker}_Close", title=f"{stock_ticker} Closing Price")
    st.plotly_chart(fig1)

# Display the volume plot in the second column
with col2:
    fig2 = px.histogram(cleaned_data, x=cleaned_data.index, y=f"{stock_ticker}_Volume", title=f"{stock_ticker} Volume")
    st.plotly_chart(fig2)

# Debug: Print the cleaned data
st.write("Details")
col1, col2 = st.columns(2)

with col1:
    st.dataframe(cleaned_data)

with col2:
    st.write("Summary Statistics")
    st.write(cleaned_data.describe())


# Check if filtered data is empty
if filtered_data.empty:
    st.warning("No data available for the selected date range. Please select a valid date range.")
else:
    # Convert dates to datetime format with timezone
    filtered_data["Datetime"] = pd.to_datetime(filtered_data.index, utc=True)
    
    # Create a two-column layout for the plots
    fig_col1, fig_col2 = st.columns(2)
    
    # Display the closing price plot in the first column
    with fig_col1:
        st.markdown("#### Closing Price Over Time")
        fig = px.line(filtered_data, x="Datetime", y=f"{stock_ticker}_Close")
        st.write(fig)

    # Display the volume plot in the second column
    with fig_col2:
        st.markdown("#### Trading Volume Over Time")
        fig = px.histogram(filtered_data, x="Datetime", y=f"{stock_ticker}_Volume")
        st.write(fig)


    # Create a two-column layout for the data frame and summary statistics
    col1, col2 = st.columns(2)
    
    # Display the cleaned data in the first column
    with col1:
        st.write(f"**{stock_ticker} - {data_interval} Detailed Data**")
        st.dataframe(filtered_data)

    # Display the summary statistics in the second column
    with col2:
        st.write("Summary Statistics")
        st.write(filtered_data.describe())
    st.success("Data loaded successfully!")   



# Function to perform feature engineering and save the model
def setup_forecast_range(data_interval):
    if data_interval == "Short Range":
        return st.sidebar.slider("Forecast Range (minutes)", min_value=5, max_value=30, value=15, step=10)
    elif data_interval == "Medium Range":
        return st.sidebar.slider("Forecast Range (hours)", min_value=10, max_value=50, value=30, step=10)
    elif data_interval == "Long Range":
        return st.sidebar.slider("Forecast Range (days)", min_value=30, max_value=365, value=365, step=30)
    else:
        return None

# Sidebar options for seasonality
st.sidebar.markdown("### Seasonality Options")
yearly_seasonality = st.sidebar.radio("Yearly Seasonality", ["On", "Off"], index=1)
weekly_seasonality = st.sidebar.radio("Weekly Seasonality", ["On", "Off"], index=0)
daily_seasonality = st.sidebar.radio("Daily Seasonality", ["On", "Off"], index=0)


# Map "On" and "Off" to True and False
yearly_seasonality = True if yearly_seasonality == "On" else False
weekly_seasonality = True if weekly_seasonality == "On" else False
daily_seasonality = True if daily_seasonality == "On" else False

# Set up the forecast range input box
forecast_range = setup_forecast_range(data_interval)
st.sidebar.write(f"Forecast Range: {forecast_range}")


if st.button("Run Stock Prediction Model"):
    #Define ticker range type
    if data_interval == "Short Range":
        ticker_range_type = "minute"
    elif data_interval == "Medium Range":
        ticker_range_type = "hour"
    elif data_interval == "Long Range":
        ticker_range_type = "day"
    else:
        ticker_range_type = None
  
    eature_engineered_data = spf.feature_engineering_and_save_data(filtered_data, ticker_range_type, ema_span, bb_window, macd_fast, macd_slow, atr_window)
    
    prophet_model, forecast = run_prophet(feature_engineered_data, periods=forecast_range, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
    
    # Debug: Print and display the forecast DataFrame
    st.write("Forecast DataFrame:")
    st.write(forecast.head())

    # Create a plotly figure with lower and upper bounds
    fig = go.Figure()
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
    y_true = filtered_data[f"{stock_ticker}_Close"][-forecast_range:]
    y_pred = forecast['yhat'][-forecast_range:]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Display evaluation metrics
    st.markdown("### Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<p style='color:black;'>RMSE: <span style='color:green;'>{rmse:.2f}</span></p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='color:black;'>MAE: <span style='color:green;'>{mae:.2f}</span></p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p style='color:black;'>MAPE: <span style='color:green;'>{mape:.2%}</span></p>", unsafe_allow_html=True)