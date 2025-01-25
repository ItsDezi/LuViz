import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from volume_plot import plot_csv  # Import the plot_csv function

with st.sidebar:
    selected = option_menu(
        menu_title="LuViz",
        options=["Home", "price_volume_plot", "Contact"],
        icons=["house", "book", "envelope"],
        menu_icon="bi-brightness-alt-high",
        default_index=0,
    )

if selected == "Home":
    st.title(f"You Have selected {selected}")

if selected == "price_volume_plot":
    st.title(f"Price and Volume comparison")
    file_path1 = 'data/TrainingData/Period1/A'  # Replace with your CSV file path
    timestamp = 'timestamp'
    askVolume = 'askVolume'
    bidVolume = 'bidVolume'
    st.plotly_chart(plot_csv(file_path1, timestamp, askVolume, bidVolume))  # Call the plot_csv function

if selected == "Contact":
    st.title(f"You Have selected {selected}")