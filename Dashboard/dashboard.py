import os
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st


# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add "src" directory to sys.path
sys.path.append(os.path.join(project_root, 'src'))
# Import prediction functions from stock_prediction_function.py
import stock_prediction_functions as spf

# Suppress all warnings
warnings.filterwarnings('ignore')

# Sample tickers
tickers = ["BRK-A", "BRK-B", "ORCL", "META", "KO", "IBM", "TSLA"]


# Set page configuration
st.set_page_config(page_title="Stock Market Prediction Dashboard", page_icon="🟢", layout="wide")
st.title("Stock Market Prediction Dashboard")


# Sidebar for user input to select stock and date range
stock_ticker = st.sidebar.selectbox('Select a Stock:', options=tickers, index=0, help="Choose the stock ticker you want to analyze.")
data_interval = st.sidebar.selectbox('Select Data Interval:', options=["short", "medium", "long"], help="Choose the data range you want.")

# display selected stock ticker and data interval
st.sidebar.markdown("### Selected Stock and Date Range")
st.sidebar.info(f"**Stock Ticker:** {stock_ticker}")
st.sidebar.info(f"**Data Interval:** {data_interval}")


# Fetch data based on user selection
if data_interval == "short":
    data = spf.get_data(stock_ticker, period='1mo', interval='2m')
elif data_interval == "medium":
    data = spf.get_data(stock_ticker, period='2y', interval='1h')
elif data_interval == "long":
    data = spf.get_data(stock_ticker, period='max', interval='1d')
else:
    st.warning("Invalid data interval selected.")


# Clean the data
cleaned_data = spf.clean_data(data)

# Checking that data was loaded correctly
if cleaned_data.empty:
    st.warning("Retry stock selection.")
else:
    st.success("Live data loaded successfully!") 
    

# Placeholder for real-time data updates
placeholder = st.empty()

# Compute KPI metrics
Daily_Returns = round(cleaned_data["Close"].pct_change().mean() * 100, 4)
Volatility = cleaned_data["Close"].pct_change().std()
Trading_Volume = cleaned_data["Volume"].mean()


# Display KPI metrics
with placeholder.container():
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Daily Returns", value=f"{Daily_Returns}%", delta=f"{Daily_Returns:.2f}%")
    col2.metric(label="Volatility", value=f"{Volatility:.2%}", delta=f"{Volatility:.2%}")
    col3.metric(label="Trading Volume", value=f"{Trading_Volume:,.0f}", delta=f"{Trading_Volume:,.0f}")

# Title for the charts
st.write(f"### {stock_ticker} Live Data")

# Check if cleaned data is empty
if cleaned_data.empty:
    st.warning("No data available for the selected date range. Please select a valid date range.")
else:
    # Convert dates to datetime format with timezone
    cleaned_data["Datetime"] = pd.to_datetime(cleaned_data.index, utc=True)
    
    # Create a two-column layout for the plots
    fig_col1, fig_col2 = st.columns(2)
    
    # Display the closing price plot in the first column
    with fig_col1:
        st.markdown("#### Closing Price Over Time")
        fig = px.line(cleaned_data, x="Datetime", y=f"Close")
        st.write(fig)

    # Display the volume plot in the second column
    with fig_col2:
        st.markdown("#### Trading Volume Over Time")
        fig = px.histogram(cleaned_data, x="Datetime", y=f"Volume")
        st.write(fig)
    
    # create a new one-column layout for candlestick plots
    new_col = st.columns(1)  # returns a list of 1 column
    with new_col[0]:  # access the first (and only) column from the list
        st.markdown("#### Candlestick Visualization")
        fig = go.Figure(data=[go.Candlestick(
                            x=cleaned_data['Datetime'],
                            open=cleaned_data['Open'],
                            high=cleaned_data['High'],
                            low=cleaned_data['Low'],
                            close=cleaned_data['Close'])])
        st.write(fig)



    # Create a two-column layout for the data frame and summary statistics
    col1, col2 = st.columns(2)
    
    # Display the cleaned data in the first column
    with col1:
        st.write(f"**{stock_ticker} - {data_interval} Detailed Data**")
        st.dataframe(cleaned_data)

    # Display the summary statistics in the second column
    with col2:
        st.write("Summary Statistics")
        st.write(cleaned_data.describe())  



# Function to perform feature engineering and save the model
def setup_forecast_range(data_interval):
    if data_interval == "short":
        return st.sidebar.slider("Forecast Range (minutes)", min_value=5, max_value=30, value=30, step=10)
    elif data_interval == "medium":
        return st.sidebar.slider("Forecast Range (hours)", min_value=10, max_value=50, value=50, step=10)
    elif data_interval == "long":
        return st.sidebar.slider("Forecast Range (days)", min_value=30, max_value=365, value=365, step=30)
    else:
        return None


# Set up the forecast range input box
forecast_range = setup_forecast_range(data_interval)
st.sidebar.write(f"Forecast Range: {forecast_range}")


if st.button("Run Stock Prediction Model"):
    #Define ticker range type
    if data_interval == "short":
        ticker_range_type = "minute"
    elif data_interval == "medium":
        ticker_range_type = "hour"
    elif data_interval == "long":
        ticker_range_type = "day"
    else:
        ticker_range_type = None
  
    feature_engineered_data = spf.feature_engineering_and_save_data(cleaned_data, save_data=False)
    
    # loading model
    # dealing with ticker names to convert them to model names
    if stock_ticker == "BRK-A":
        model_name = "BRK_A"
    if stock_ticker == "BRK-B":
        model_name = "BRK_B"
    if stock_ticker == "KO":
        model_name = "COCA"
    if stock_ticker == "IBM":
        model_name = "IBM"
    if stock_ticker == "META":
        model_name = "META"
    if stock_ticker == "ORCL":
        model_name = "ORACLE"
    if stock_ticker == "TSLA":
        model_name = "TESLA"

    model_path = project_root + f"/models/{model_name}_{data_interval}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    

    prophet_model, trans_forcast, forecast = spf.run_prophet(feature_engineered_data, 
                                                             train=False, 
                                                             cross_validate=False, 
                                                             predict=True,
                                                             prophet=model,
                                                             periods=forecast_range)
    
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
    # Make predictions on the test set
    df = feature_engineered_data[['Close', 'Volume', 'MACD', 'ATR']].reset_index()
    df.rename(columns={'Datetime': 'ds', 'Date' : 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone information

    test_data = df.iloc[int(0.8 * len(df)):]
    eval_forecast = model.predict(test_data)

    y_true = test_data['y']
    y_pred = eval_forecast['yhat']


    # Compute MSE (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)

    # Compute RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
        
    # Display evaluation metrics
    st.markdown("### Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p style='color:white;'>RMSE: <span style='color:green;'>{rmse:.2f}</span></p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='color:white;'>MSE: <span style='color:green;'>{mse:.2f}</span></p>", unsafe_allow_html=True)



    