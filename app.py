import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import os
from luviz.plotter import plot_group_price_time_series_scatter, plot_candlestick

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
    # st.plotly_chart(plot_group_price_time_series_scatter({'A'}, 1, {'trade', 'bid'}))  # Call the plot_csv function
    st.plotly_chart(plot_candlestick('A', 1, '1S'))  # Call the plot_csv function
if selected == "price_volume_plot":
    st.title(f"Price and Volume comparison")
    # st.plotly_chart(plot_group_price_time_series_scatter({'A'}, 1, {'bid'}))  # Call the plot_csv function
    # st.plotly_chart(plot_group_price_time_series_scatter({'B'}, 1, {'bid'}))  # Call the plot_csv function
    st.plotly_chart(plot_group_price_time_series_scatter({'C'}, 1, {'bid'}))  # Call the plot_csv function
    # st.plotly_chart(plot_group_price_time_series_scatter({'D'}, 1, {'bid'}))  # Call the plot_csv function
    # st.plotly_chart(plot_group_price_time_series_scatter({'E'}, 1, {'bid'}))  # Call the plot_csv function

if selected == "Contact":
    st.title(f"You Have selected {selected}")