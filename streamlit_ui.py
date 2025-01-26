import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from volume_plot import plot_csv  # Import the plot_csv function
from luviz.plotter import plot_group_price_time_series_scatter, plot_candlestick, plot_price_time_series_scatter_subplots, plot_std_dev, plot_stock_comparison_tool, plot_toggleable_volume_price

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="LuViz",
        options=["Asset Comparison", "Price Vs. Volume", "Candlestick Charts", "Standard Deviation"],
        icons=["house", "graph-up", "align-middle"],
        menu_icon="bi-brightness-alt-high",
        default_index=0,
    )

if selected == "Asset Comparison":
    st.title(f"Asset Comparison")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        option_stock = st.selectbox(
            "Stock",
            ("A", "B", "C", "D", "E"),
            key="col1"
        )
    with col2:
        option_period = st.selectbox(
            "Period",
            list(range(1, 16)),
            key="col2"
        )
    with col3:
        option_stock2 = st.selectbox(
            "Stock",
            ("A", "B", "C", "D", "E"),
            key="col3"
        )
    with col4:
        option_period2 = st.selectbox(
            "Period",
            list(range(1, 16)),
            key="col4"
        )
    st.plotly_chart(plot_stock_comparison_tool(option_stock, option_period2, option_stock2, option_period2), use_container_width=False)    


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

    tmp = plot_toggleable_volume_price(option_stock, option_period)
    #print(tmp)

    st.write(tmp)  # Call the plot_csv function
    # c1, c2 = st.columns(2)
    # with c1:

    #     st.markdown("Maximum Bid Volume: " + str())
    #     st.markdown("Maximum Ask Volume: " + str())
    #     st.markdown("Maximum Bid Price: " + str())
    #     st.markdown("Maximum Ask Price: " + str())
    #     st.markdown("Maximum Trade Price: " + str())
    # with c2:
    #     st.markdown("Minimum Bid Volume: " + str())
    #     st.markdown("Minimum Ask Volume: " + str())
    #     st.markdown("Minimum Bid Price: " + str())
    #     st.markdown("Minimum Ask Price: " + str())
    #     st.markdown("Minimum Trade Price: " + str())


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

if selected == "Standard Deviation":
    st.title("Standard Deviation")
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
            "Interval",
            ("30S", "60S")
        )
    file_path1 = 'data/TrainingData' + '/Period' + str(option_period) + '/Period' + str(option_period) + '/' + str(option_stock)  # Replace with your CSV file path
    st.plotly_chart(plot_std_dev(option_stock, option_period, option_frequency))
