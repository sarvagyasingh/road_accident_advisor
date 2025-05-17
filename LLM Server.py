from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

# === Configuration ===
GGUF_MODEL_PATH = "/Users/sarvagya/PycharmProjects/Streamlit_HAB/gemma-3-finetune.Q8_0.gguf"  # replace with your actual path
PORT = 8000

# === Initialize Model ===
print("Loading model... This may take a moment.")
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() or 4,
    use_mlock=True  # optional, can help with performance
)
print("Model loaded successfully.")

# === Flask App ===
app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field in JSON"}), 400

    output = llm(prompt, max_tokens=256, stop=["</s>"])
    return jsonify({
        "prompt": prompt,
        "response": output["choices"][0]["text"].strip()
    })

@app.route("/", methods=["GET"])
def health():
    return "🦙 LLM Inference API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)