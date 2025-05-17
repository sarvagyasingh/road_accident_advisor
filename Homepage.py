import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Crash Insight AI", layout="wide")

st.title("🚧 Crash Insight AI")

st.markdown("""

**Your smart assistant for understanding and summarizing road accidents.**

This tool leverages machine learning and a fine-tuned language model to:

- 🧠 **Predict crash severity** from real-world accident data  
- ✍️ **Generate natural-language summaries** from structured crash reports  
- 💬 **Chat with the AI** to get insights, precautions, or crash scenario analysis

Use the tabs above to explore real data, run predictions, and chat with the model.
""")