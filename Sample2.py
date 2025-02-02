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


import streamlit as st #web development
import numpy as np #np mean, np random 
import pandas as pd #pd read_csv, df manipulation
import matplotlib.pyplot as plt #plotting
import plotly.express as px #interactive charts
import time



# Sample tickers
tickers = ["BRK-A", "BRK-B", "ORCL", "META", "KO", "IBM", "TSLA"]



#read  csv from github repository
url_BRK_A_medium = pd.read_csv("https://raw.githubusercontent.com/Natnael-Getahun/Stock-Market-Predictions/main/Datasets/BRK_A_medium.csv")

st.set_page_config (page_title= "Stock Market Prediction Dashboard", page_icon= "🟢", layout="wide")

#Dashboard Title
st.title("Stock Market Prediction Dashboard")

# Allow the user to choose a stock
stock_ticker = st.sidebar.selectbox('Select a Stock:', tickers)

# Set up the sidebar with start and end date input boxes
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Display the selected dates
st.sidebar.write(f"Start Date: {start_date}")
st.sidebar.write(f"End Date: {end_date}")

# Load stock data (for demonstration purposes, using random data)
data = pd.DataFrame({
    "Datetime": pd.date_range(start="2022-01-01", periods=17520, freq="H"),
    "Close": np.random.randint(100, 500, 17520),
    "Volume": np.random.randint(100000, 1000000, 17520)
})

# Convert start_date and end_date to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

#filter data based on the selected dates
filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]


# Display the filtered data(might be necessary)
#st.sidebar.write("Filtered Data:")


#Creating a single element container 
placeholder = st.empty()

# function to fetch and simulate real-time/live data simulation 
def fetch_live_data(stock_filter):
#wile loop to simulate real-time data
#while True:
    return pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=17520, freq="H"),
        "close": np.random.randint(100, 500, 17520),
        "volume": np.random.randint(100000, 1000000, 17520)
    })
      
#kpi metrics note 
Daily_Returns = round(url_BRK_A_medium["Close"].pct_change().mean()*100,4) # Multiply by 100 to get percentage and round to 2 decimal places
Volatility = url_BRK_A_medium["Close"].pct_change().std()
Trading_Volume = url_BRK_A_medium["Volume"].mean()


#fill in the threecolumns with the kpi metrics
with placeholder.container():
    # Define layout columns (avoid name conflicts)
    col1, col2, col3 = st.columns(3)

    # Display metrics using actual data values (replace with correct variables)
    col1.metric(label="Daily Returns", value=f"{Daily_Returns}", delta=f"{Daily_Returns:.2%}")
    col2.metric(label="Volatility", value=f"{Volatility:.2%}", delta=f"{Volatility:.2%}")
    col3.metric(label="Trading Volume", value=f"{Trading_Volume:,.0f}", delta=f"{Trading_Volume:,.0f}")
#create 2 columns for the charts
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("#### Closing Price Over Time")
        fig = px.line(url_BRK_A_medium, x="Datetime", y="Close")
        st.write(fig)
    with fig_col2:
        st.markdown("#### Trading Volume Over Time")
        fig = px.histogram(url_BRK_A_medium, x="Datetime", y="Volume")
        st.write(fig)
    st.markdown("#### Detailed data view")
    st.dataframe(url_BRK_A_medium) #display the dataframe
    time.sleep(0.5) #sleep for 0.5 second before fetching new data

#function to fetch and simulate real-time/live data simulation
#def stock_selection (stock_filter):

#creating pridiction models for the selected stock

