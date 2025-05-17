import streamlit as st
import requests

st.subheader("💬 Chat with Crash Insight AI")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
with st.form("chat_form", clear_on_submit=True):
    user_message = st.text_input("You:", placeholder="Ask anything about crash scenarios or safety tips...")
    submitted = st.form_submit_button("Send")

if submitted and user_message:
    # Add user message
    st.session_state.chat_history.append(("You", user_message))

    # Send to Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "safety-advisor",
                "prompt": f"User: {user_message}\nAssistant:",
                "stream": False,
                "options": {
                    "num_predict": 512,
                    "stop": ["User:", "You:", "🤖"]
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        bot_message = result.get("response", "I'm sorry, I couldn't process that.")
    except Exception as e:
        bot_message = f"(Error from model: {e})"

    # Add bot response
    st.session_state.chat_history.append(("Crash Insight AI", bot_message))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧍 You:** {msg}")
    else:
        st.markdown(f"**🤖 Crash Insight AI:** {msg}")