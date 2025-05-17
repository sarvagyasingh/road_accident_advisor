import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Crash Insight AI", layout="wide")

st.title("🚧 Crash Insight AI")

st.markdown("""
Welcome to **Crash Insight AI**, your smart assistant for analyzing and understanding road accidents.
This app uses machine learning to predict crash severity and summarizes accidents using a fine-tuned language model trained on accident data.

➡️ Go to the **Application** tab to get started.
""")