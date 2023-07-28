import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
import yfinance as yf



#streamlit run main.py (to run the code)

start = '2017-01-01'
end='2023-05-01'

time_start= datetime.strptime(start,'%Y-%m-%d').date()
time_end= datetime.strptime(end,'%Y-%m-%d').date()

tab1, tab2 = st.tabs(["stock and predicted graph", "predicted prices"])

with tab1:
    st.title('Stock trend prediction')
    user_input = st.text_input('Enter Stock Ticker','GOOG')
    df= yf.download(user_input, time_start,time_end)#scrapping the data
    # Describing the data
    st.subheader('Description of dataset from 2017-2023')
    st.write(df.describe())
    # VISUALISATIONS
    # plot closing price chart
    st.subheader('Closing Price vs Time chart--- ' + user_input)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # spliting data into training and testing
    # what we usually do for data predtictions

    data_training = pd.DataFrame(
        df['Close'][0:int(len(df) * 0.70)])  # 70% of data in training whislt the rest is in testing
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])


    # scaling down the data


    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # load my model

    model = load_model('keras_model.h5')

    # testing part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):  # 555
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]  # at initial postion

    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final predicted graph
    st.subheader('Predictions vs original for ' + user_input + ' stock dataset')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original close price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

with tab2:
    st.title("predictions")
    original_data = y_test.tolist()
    predicted_data = y_predicted.tolist()
    df2 = pd.DataFrame(list(zip(original_data, predicted_data)), columns=['Closing_original_prices', 'Predicted_prices'])
    st.subheader(user_input + " dataset")
    st.write(df.head())
    st.subheader('closing vs predicted prices for ' + user_input + ' dataset')
    st.write(df2.head(10))
















