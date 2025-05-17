import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Crash Insight AI", layout="wide")

st.title("🚧 Crash Insight AI")
st.markdown("""
Welcome to **Crash Insight AI**, your smart assistant for analyzing and understanding road accidents.
This app uses machine learning to predict crash severity and summarizes accidents using a fine-tuned language model trained on accident data.

➡️ Go to the **Predict & Summarize** tab to get started.
""")

# Load data from dataset.csv
full_df = pd.read_csv("dataset.csv")
df = full_df.head(500)

st.subheader("🔍 Crash Instance")

# Initialize session state for current row index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = ""

# Get the current row data
current_row = df.loc[st.session_state.current_index]

def generate_prompt_completion(row):
    weather = row.get('weather_description', 'unknown')
    day = row.get('dayofweek', 'unknown')
    body = row.get('body_description', 'unknown')
    segment = row.get('segment_id', 'unknown')
    severity = row.get('severity_description', 'unknown')
    prompt = (
        f"The crash occurred on a {day} with weather described as {weather}.\n"
        f"The vehicle body type involved was {body}.\n"
        f"The location segment ID is {segment}.\n"
        f"The reported severity description is {severity}.\n"
        f"Please summarize this crash data and predict the severity."
    )
    return prompt

# Display essential fields of the current row in a vertical, card-style format with icons and readable labels
fields_to_display = [
    {
        "field": "weather_description",
        "label": "Weather",
        "icon": "☁️"
    },
    {
        "field": "dayofweek",
        "label": "Day of Week",
        "icon": "📅"
    },
    {
        "field": "body_description",
        "label": "Vehicle Body Type",
        "icon": "🚗"
    },
    {
        "field": "segment_id",
        "label": "Location Segment ID",
        "icon": "🗺️"
    },
    {
        "field": "severity_description",
        "label": "Severity",
        "icon": "⚠️"
    },
    {
        "field": "direction_description_before_crash",
        "label": "Direction Before Crash",
        "icon": "↗️"
    },
    {
        "field": "roadwaytype_description",
        "label": "Road Type",
        "icon": "🛣️"
    },
    {
        "field": "agencyidentifier",
        "label": "Reporting Agency",
        "icon": "🏢"
    },
]

row_display = {f["field"]: current_row.get(f["field"], 'unknown') for f in fields_to_display}

with st.container():
    st.markdown(f"### Row {st.session_state.current_index + 1} of {len(df)}")
    col1, col2 = st.columns(2)
    half = len(fields_to_display) // 2
    fields_col1 = fields_to_display[:half]
    fields_col2 = fields_to_display[half:]
    with col1:
        card_html_parts = [
            '<div style="background: #f6f6fa; border-radius: 1rem; padding: 1.5rem 1.5rem 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem; max-width: 420px;">'
        ]
        for field_info in fields_col1:
            icon = field_info["icon"]
            label = field_info["label"]
            value = row_display[field_info["field"]]
            card_html_parts.append(
                '<div style="display: flex; align-items: center; margin-bottom: 0.85rem;">'
                f'<span style="font-size:1.5rem; margin-right: 0.7rem;">{icon}</span>'
                f'<span style="font-weight: 600; font-size: 1.08rem; color: #222;">{label}:</span>'
                f'<span style="margin-left: 0.5rem; font-size:1.07rem; color:#333;">{value}</span>'
                '</div>'
            )
        card_html_parts.append('</div>')
        card_html = ''.join(card_html_parts)
        st.markdown(card_html, unsafe_allow_html=True)
    with col2:
        card_html_parts = [
            '<div style="background: #f6f6fa; border-radius: 1rem; padding: 1.5rem 1.5rem 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem; max-width: 420px;">'
        ]
        for field_info in fields_col2:
            icon = field_info["icon"]
            label = field_info["label"]
            value = row_display[field_info["field"]]
            card_html_parts.append(
                '<div style="display: flex; align-items: center; margin-bottom: 0.85rem;">'
                f'<span style="font-size:1.5rem; margin-right: 0.7rem;">{icon}</span>'
                f'<span style="font-weight: 600; font-size: 1.08rem; color: #222;">{label}:</span>'
                f'<span style="margin-left: 0.5rem; font-size:1.07rem; color:#333;">{value}</span>'
                '</div>'
            )
        card_html_parts.append('</div>')
        card_html = ''.join(card_html_parts)
        st.markdown(card_html, unsafe_allow_html=True)

with st.container():
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col2:
        if st.button("Run Inference", key="run_inference_btn"):
            try:
                # Replace NaN and infinite values with None before sending
                cleaned_row = current_row.replace([np.nan, np.inf, -np.inf], None)
                # Generate prompt using the new function
                prompt = generate_prompt_completion(cleaned_row)
                # Use Ollama endpoint for inference
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "safety-advisor", "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.llm_summary = result.get("response", "No summary available.")
            except Exception as e:
                st.warning(f"Failed to get prediction from backend: {e}")
                st.session_state.llm_summary = "No summary available."
    with nav_col3:
        import random
        if st.button("Next Row", key="next_row_btn"):
            # Ensure a different index than the current one
            new_index = st.session_state.current_index
            while new_index == st.session_state.current_index:
                new_index = random.randint(0, 499)
            st.session_state.current_index = new_index
            # Do not clear llm_summary here to preserve last model summary



if st.session_state.llm_summary:
    st.markdown("---")
    st.subheader("Crash Summary from Model")
    st.markdown(st.session_state.llm_summary)
