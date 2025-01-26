import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from volume_plot import plot_csv  # Import the plot_csv function
from luviz.plotter import plot_group_price_time_series_scatter

with st.sidebar:
    selected = option_menu(
        menu_title="LuViz",
        options=["Home", "Price Vs. Volume", "Contact"],
        icons=["house", "graph-up", "envelope"],
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
            ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15")
        )
    file_path1 = 'data/TrainingData/Period' + option_period + '/' + option_stock  # Replace with your CSV file path


    st.write(plot_csv(file_path1, 'timestamp', 'askVolume', 'bidVolume'))  # Call the plot_csv function
    st.button("Add a chart", type="primary")


if selected == "Contact":
    st.title(f"You Have selected {selected}")