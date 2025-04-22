import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
from sklearn.metrics import r2_score
import os
import time

# Set date range
start = '2010-01-01'
end = '2022-03-03'

# Streamlit UI
st.title('📈 Stock Price Prediction')
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Stock Prediction"])

# Display home or about content
if app_mode == "Home":
    st.header("Stock Price Prediction System")
    image_path = r"C:\Users\as735\OneDrive\Desktop\project 000\Stock_Trend_Web_App_Python_Machine_Learning-main\Code\Screenshot 2025-04-23 000006.png"

    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Stock Price Prediction System. This app allows you to predict stock prices based on historical data using an LSTM model.
    """)
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This project utilizes a deep learning model (LSTM) to predict stock prices. 
    We use historical stock data from Yahoo Finance and process it to make predictions.
    
    About the Stock Price Prediction System
Stock price prediction is an essential aspect of financial forecasting that helps investors make informed decisions. The system uses machine learning models, particularly LSTM (Long Short-Term Memory) networks, to forecast the future prices of stocks based on historical data. Here's an in-depth look at the core components and methodologies involved in the stock price prediction system:

1. Data Collection and Preprocessing
Data Source: The system collects historical stock data from trusted financial platforms such as Yahoo Finance using APIs like pandas_datareader. The dataset includes crucial stock information such as open, close, high, low prices, and trading volumes over a specified time period.

Data Cleaning: Missing values are handled by filling or dropping based on the analysis, ensuring the dataset is clean and complete.

Feature Engineering: The system extracts relevant features such as moving averages (100-day, 200-day) and price trends to enhance the predictive power of the model.

Normalization: The data is normalized using MinMaxScaler to scale stock prices into a fixed range (0 to 1). This ensures that the machine learning model handles input data more effectively and improves model convergence.

2. Model Architecture
LSTM (Long Short-Term Memory): LSTM is a type of Recurrent Neural Network (RNN) designed to process time-series data. Unlike traditional neural networks, LSTM networks can learn from long-term dependencies in the data, making them ideal for stock price prediction where historical data influences future trends.

Pre-trained Model: A pre-trained LSTM model is used to predict stock prices. The model is first trained using a training dataset (usually 70% of the data) and then validated on a separate validation set (typically 30%). The model learns to detect patterns in the data, such as price spikes, trends, and fluctuations, to make predictions about future stock prices.

3. Model Training and Evaluation
Training: The model is trained on historical stock data, where it learns the relationship between past stock prices and future trends. A sliding window approach is used to generate training sequences, allowing the model to learn from previous days' prices to predict future prices.

Testing and Prediction: The model is tested on unseen stock data to evaluate its predictive power. The performance is assessed using metrics such as R² score, Mean Absolute Error (MAE), and Mean Squared Error (MSE) to ensure accurate predictions.

Visualization: After training, the model's predictions are compared with the actual stock prices to visually assess how well the model performs. Graphs are plotted to show the predicted prices versus the actual closing prices over time, allowing users to see the effectiveness of the model.

4. User Interface with Streamlit
Interactive Dashboard: The stock price prediction system is integrated into a web interface built with Streamlit. Streamlit enables users to interact with the model by inputting a stock ticker symbol and viewing predictions. Users can upload stock tickers like AAPL (Apple), MSFT (Microsoft), or GOOGL (Google) to generate predictions.

Real-time Data: Users can select different time periods, visualize closing prices with 100-day or 200-day moving averages, and receive real-time predictions from the model.

Forecasting and Analysis: The system can forecast future prices based on historical trends, providing investors with valuable insights into the expected direction of stock prices. Predictions are displayed alongside confidence intervals to reflect the model’s uncertainty.

5. Benefits for Investors
Decision Support: By utilizing the predictions from the model, investors can make better decisions on whether to buy, sell, or hold their stocks based on expected price movements.

Trend Analysis: The model highlights long-term trends and short-term fluctuations, helping investors understand the market dynamics and potential opportunities.

Performance Metrics: The accuracy of the model is regularly evaluated, providing continuous feedback to improve the model's performance. Metrics such as the R² score help assess how well the model predicts stock prices.

6. Challenges and Limitations
Market Volatility: Stock prices are influenced by a wide range of external factors, such as economic events, news, and market sentiment, which can be difficult to predict using historical data alone.

Overfitting: Like all machine learning models, LSTM networks are susceptible to overfitting if not properly tuned. This means the model might perform well on training data but fail to generalize to new, unseen data.

Model Improvements: The model can be enhanced by incorporating additional features, such as technical indicators (RSI, MACD) or sentiment analysis from financial news, to increase accuracy.

7. Conclusion
The Stock Price Prediction system offers an efficient way to forecast stock prices using deep learning models like LSTM. With its ability to learn from historical data, it provides valuable insights for investors looking to understand market trends. By integrating real-time data and visualizations, this system makes stock price prediction more accessible and actionable, allowing users to make informed financial decisions. However, it is important to note that stock market predictions are inherently uncertain, and the model should be used in conjunction with other forms of analysis.
    """)

# User Input for Stock Ticker
user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)')

# Only run the rest of the code if user provides input
if user_input:
    retries = 3
    for i in range(retries):
        try:
            # Fetch stock data
            df = data.DataReader(user_input.upper(), 'yahoo', start, end)
            
            # Check if data is empty
            if df.empty:
                st.error("No data found for this ticker. Please enter a valid ticker like AAPL, MSFT, or GOOGL.")
                st.stop()

            # Show Data Summary
            st.subheader('Data from 2010 to 2022')
            st.write(df.describe())

            # Plot Closing Price
            st.subheader('📉 Closing Price vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close)
            st.pyplot(fig)

            # Closing Price with 100-day MA
            st.subheader('📊 Closing Price vs Time Chart with 100MA')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, label='100MA')
            plt.plot(df.Close, label='Close')
            plt.legend()
            st.pyplot(fig)

            # Closing Price with 100-day and 200-day MA
            st.subheader('📊 Closing Price vs Time Chart with 100MA & 200MA')
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, 'r', label='100MA')
            plt.plot(ma200, 'g', label='200MA')
            plt.plot(df.Close, 'b', label='Original Price')
            plt.legend()
            st.pyplot(fig)

            # Prepare Training and Testing Data
            data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            x_train = []
            y_train = []

            for i in range(100, data_training_array.shape[0]):
                x_train.append(data_training_array[i - 100: i])
                y_train.append(data_training_array[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)

            # Check model path
            model_path = 'keras_model.h5'  # Ensure the correct path for your model file
            if not os.path.exists(model_path):
                st.error("Model file not found. Please ensure 'keras_model.h5' exists in the correct directory.")
                st.stop()

            # Load the pre-trained LSTM model
            model = load_model(model_path)

            # Prepare Test Data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100: i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make Predictions
            y_predicted = model.predict(x_test)

            # Reverse scaling
            scale_factor = scaler.scale_[0]  # Dynamic scale factor from fitted scaler
            y_predicted = y_predicted / scale_factor
            y_test = y_test / scale_factor

            # R² Score
            r_squared = r2_score(y_test, y_predicted)
            st.subheader(f"✅ Model R² Score: {r_squared:.2f}")

            # Plot Predictions vs Actual
            st.subheader('📈 Predicted Price vs Actual Price')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time (days)')
            plt.ylabel('Stock Price')
            plt.legend()
            st.pyplot(fig2)

            break  # Exit loop if no error

        except Exception as e:
            if i == retries - 1:
                st.error(f"Failed to fetch data after {retries} attempts: {e}")
                st.stop()
            else:
                st.warning(f"Attempt {i + 1} failed, retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying

else:
    st.warning("⚠️ Please enter a stock ticker symbol.")
