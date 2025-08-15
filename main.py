from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import tempfile
import mimetypes
import time
import requests
import json

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Data Analyst Agent is live. Use POST /api/ with a question file."

@app.route("/api/", methods=["POST"])
def handle_request():
    start_time = time.time()

    if "questions.txt" not in request.files:
        return jsonify({"error": "questions.txt file is required"}), 400

    question_text = request.files["questions.txt"].read().decode("utf-8")

    # Save uploaded files (except questions.txt)
    file_data = {}
    for key in request.files:
        if key == "questions.txt":
            continue
        file = request.files[key]
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        file_data[file.filename] = tmp_path

    try:
        # 1. Prepare prompt and system message
        system_prompt = """You are a data analyst assistant. Given a task description and uploaded files, your job is to:
- Decide what operations (analysis, web scraping, visualization, etc.) are needed.
- Extract, clean, analyze, and visualize the data.
- Generate results in the required format (JSON, base64 plots, etc.).
- If the question asks for plots, encode them as base64 image URIs under 100,000 bytes.
- Return structured data only, like a JSON array or object, depending on the prompt.
- Do not explain your reasoning, just return the answer."""

        file_summary = "\n".join([f"- {k}: {mimetypes.guess_type(k)[0] or 'Unknown type'}" for k in file_data])

        user_prompt = f"""Task:\n{question_text}

Uploaded Files:
{file_summary}

Please return ONLY the final result in valid JSON (no extra explanation)."""

        # 2. Get the AIPipe token from environment or config
        aipipe_token = os.environ.get("AIPIPE_TOKEN")
        if not aipipe_token:
            return jsonify({"error": "AIPIPE_TOKEN is not set in environment variables"}), 401

        # 3. Send request to AIPipe endpoint
        headers = {
            "Authorization": f"Bearer {aipipe_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        llm_response = requests.post(
            "https://aipipe.org/openrouter/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )

        if llm_response.status_code != 200:
            return jsonify({"error": "Failed to fetch from GPT", "details": llm_response.text}), 500

        response_data = llm_response.json()
        final_answer = response_data["choices"][0]["message"]["content"]

        # Enforce time limit
        if time.time() - start_time > 170:
            return jsonify({"error": "Request took too long"}), 500

        return final_answer, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
