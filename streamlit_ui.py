import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from volume_plot import plot_csv  # Import the plot_csv function
from luviz.plotter import plot_group_price_time_series_scatter, plot_candlestick, plot_price_time_series_scatter_subplots, plot_toggleable_volume_price

with st.sidebar:
    selected = option_menu(
        menu_title="LuViz",
        options=["Home", "Price Vs. Volume", "Candlestick Charts"],
        icons=["house", "graph-up", "align-middle"],
        menu_icon="bi-brightness-alt-high",
        default_index=0,
    )

if selected == "Home":
    st.title(f"You Have selected {selected}")

if selected == "Price Vs. Volume":
    st.title(f"Price and Volume comparison")
    select_param = st.container()

    col1, col2 = st.columns(2)
    with col1:
        option_stock = st.selectbox(
            "Stock",
            ("A", "B", "C", "D", "E"),
        )
    with col2:
        option_period = st.selectbox(
            "Period",
            list(range(1, 16))
        )
    file_path1 = 'data/TrainingData' + '/Period' + str(option_period) + '/Period' + str(option_period) + '/' + option_stock  # Replace with your CSV file path


    st.write(plot_toggleable_volume_price(option_stock, option_period))  # Call the plot_csv function
    

if selected == "Candlestick Charts":
    st.title(f"{selected}")
    select_param = st.container()

    col1, col2, col3 = st.columns(3)
    with col1:
        option_stock = st.selectbox(
            "Stock",
            ("A", "B", "C", "D", "E"),
        )
    with col2:
        option_period = st.selectbox(
            "Period",
            list(range(1, 16))
        )
    with col3:
        option_frequency = st.selectbox(
            "Frequency",
            ("1s", "5s", "10s", "15s", "30s", "60s")
        )
    file_path1 = 'data/TrainingData' + '/Period' + str(option_period) + '/Period' + str(option_period) + '/' + str(option_stock)  # Replace with your CSV file path


    st.plotly_chart(plot_candlestick(option_stock, option_period, option_frequency))