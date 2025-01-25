import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu


with st.sidebar:
  selected = option_menu(
    menu_title = "LuViz",
    options = ["Home","Projects","Contact"],
    icons = ["house","book","envelope"],
    menu_icon = "bi-brightness-alt-high",
    default_index = 0,
  )
if selected == "Home":
    st.title(f"You Have selected {selected}")
if selected == "Projects":
    st.title(f"You Have selected {selected}")
if selected == "Contact":
    st.title(f"You Have selected {selected}")